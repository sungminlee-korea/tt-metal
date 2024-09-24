// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_elffile.hpp"

#include <cstdarg>
#include <type_traits>

#include "common/assert.hpp"
// C
#include <errno.h>
// OS
#include <elf.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// Verify some knowledge of, and compatibilty with, RiscV
#ifndef EM_RISCV
#error "Don't know RISCV elf details"
#endif

// Having the same endianness as RISCV makes things easier.
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "Host must be little endian"
#endif

#ifdef PT_RISCV_ATTRIBUTES
#warning "PT_RISCV_ATTRIBUTES available, remove workaround"
#else
// Missing from my elf.h
#define PT_RISCV_ATTRIBUTES (PT_LOPROC + 3)
enum {
  Tag_RISCV_arch = 5,
};
#endif

// Sadly the toolchain's usurped some machine numbers.  With any luck
// this will go away at some point.
#define EM_RISCV_GRAYSKULL 242
#define EM_RISCV_WORMHOLE  0x5151
#define EM_RISCV_BLACKHOLE 0x6151

// We have to translate these two instructions
static constexpr uint32_t insn_opc_auipc = 0x00000017;
static constexpr uint32_t insn_opc_lui = 0x00000037;
static constexpr uint32_t insn_mask_u = 0x0000007f;
static constexpr uint32_t mask_hi20 = 0x00000fff;
static constexpr uint32_t mask_hi20_shift = 12;
static constexpr uint32_t mask_lo12_i = 0x000fffff;
static constexpr uint32_t mask_lo12_i_shift = 20;
static constexpr uint32_t mask_lo12_s = 0x01fff07f;
static constexpr uint32_t mask_lo12_s_split = 5;
static constexpr uint32_t mask_lo12_s_shift_1 = 7;
static constexpr uint32_t mask_lo12_s_shift_2 = 25;

using namespace ll_api;

class ElfFile::Impl {
private:
  std::span<Elf32_Phdr const> Phdrs;
  std::span<Elf32_Shdr const> Shdrs;
  std::span<std::byte const> StrTab;
  char const *Arch = nullptr;
  std::string const &Path;

private:
  ElfFile &Owner;

public:
  Impl (ElfFile &owner, std::string const &path)
    : Owner(owner), Path(path) {}
  ~Impl () = default;

public:
  void loadImage ();
  void relocateImage ();

private:
  Elf32_Ehdr const &getHeader () const {
    return *reinterpret_cast<Elf32_Ehdr const *>(Owner.Contents.data());
  }
  std::span<Elf32_Phdr const> getPhdrs () const { return Phdrs; }
  std::span<Elf32_Shdr const> getShdrs () const { return Shdrs; }
  std::span<std::byte> getContents (Elf32_Phdr const &phdr) {
    return Owner.Contents.subspan(phdr.p_offset, phdr.p_filesz);
  }
  std::span<std::byte> getContents (Elf32_Shdr const &shdr) {
    return Owner.Contents.subspan(shdr.sh_offset, shdr.sh_size);
  }
  char const *getString (size_t offset) {
    if (offset < StrTab.size())
      return byteOffset<char const>(StrTab.data(), offset);
    else
      return "???";
  }
  char const *getName (Elf32_Shdr const &shdr) {
    return getString(shdr.sh_name);
  }

private:
  std::span<Elf32_Sym const> getSymbols (Elf32_Shdr const &shdr) {
    auto section = getContents(shdr);
    return std::span(byteOffset<Elf32_Sym const>(section.data()),
                     section.size() / shdr.sh_entsize);
  }
  char const *getName (Elf32_Sym const &sym) { return getString(sym.st_name); }
  std::span<Elf32_Rela> getRelocations (Elf32_Shdr const &shdr) {
    auto section = getContents(shdr);
    return std::span(byteOffset<Elf32_Rela>(section.data()),
                     section.size() / shdr.sh_entsize);
  }

private:
  char const *getArchFromAttrs(std::span<std::byte>);

private:
  template <typename T = std::byte>
  static T *byteOffset (std::byte *base, size_t offset = 0) {
    return reinterpret_cast<T *>(base + offset);
  }
  template <typename T = std::byte>
  static T const *byteOffset (std::byte const *base, size_t offset = 0) {
    return reinterpret_cast<T const *>(base + offset);
  }

  static uint32_t read32 (std::span<std::byte> contents, uint32_t offset) {
    return *byteOffset<uint32_t>(contents.data(), offset);
  }
  static void write32 (std::span<std::byte> contents, uint32_t offset,
                       uint32_t value) {
    *byteOffset<uint32_t>(contents.data(), offset) = value;
  }
};

ElfFile::ElfFile (std::string const &path) {
  int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  struct stat st;
  void *buffer = MAP_FAILED;
  if (fd >= 0 && fstat(fd, &st) >= 0)
    buffer = mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd,
                  0);
  if (fd >= 0)
    // It is acceptable to close a mapped file -- the mapping stays.
    close(fd);
  if (buffer == MAP_FAILED)
    TT_THROW("{}: cannot map elf file into memory: {}", path, strerror(errno));

  Contents = std::span(reinterpret_cast<std::byte *>(buffer), st.st_size);

  Impl impl(*this, path);

  impl.loadImage();
  // FIXME: impl.relocateImage();
}

ElfFile::~ElfFile () {
  if (!Contents.empty())
    munmap(Contents.data(), Contents.size());
}

void ElfFile::Impl::loadImage () {
  auto &hdr = getHeader();

  // Make sure it's ELF
  if (hdr.e_ident[EI_MAG0] != 0x7f || hdr.e_ident[EI_MAG1] != 'E'
      || hdr.e_ident[EI_MAG2] != 'L' || hdr.e_ident[EI_MAG3] != 'F')
    TT_THROW("{}: no ELF magic found", Path);

  // Of the expected address size, endianness and version
  if (hdr.e_ident[EI_CLASS] != ELFCLASS32
      || hdr.e_ident[EI_DATA] != ELFDATA2LSB
      || hdr.e_ident[EI_VERSION] != EV_CURRENT)
    TT_THROW("{}: incompatible address size or endianness", Path);

  if (hdr.e_type != ET_EXEC)
    TT_THROW("{}: not an executable", Path);

  if (hdr.e_machine != EM_RISCV
      // Hopefully these can go way at some point.
      && hdr.e_machine != EM_RISCV_GRAYSKULL
      && hdr.e_machine != EM_RISCV_WORMHOLE
      && hdr.e_machine != EM_RISCV_BLACKHOLE)
    TT_THROW("{}: incompatible architecture {}", Path, hdr.e_machine);

  if (!hdr.e_phoff || hdr.e_phoff & 3 || hdr.e_phentsize != sizeof(Elf32_Phdr)
      || (hdr.e_phoff + hdr.e_phnum * sizeof(Elf32_Phdr)
          > Owner.Contents.size()))
    TT_THROW("{}: PHDRS are missing or malformed", Path);
  Phdrs = std::span(
      byteOffset<Elf32_Phdr const>(Owner.Contents.data(), hdr.e_phoff),
      hdr.e_phnum);
  if (!hdr.e_shoff || hdr.e_shoff & 3 || hdr.e_shentsize != sizeof(Elf32_Shdr)
      || (hdr.e_shoff + hdr.e_shnum * sizeof(Elf32_Shdr)
          > Owner.Contents.size()))
    TT_THROW("{}: sections are missing or malformed", Path);
  Shdrs = std::span(
      byteOffset<Elf32_Shdr const>(Owner.Contents.data(), hdr.e_shoff),
      hdr.e_shnum);
  if (!hdr.e_shstrndx || hdr.e_shstrndx >= Shdrs.size())
    TT_THROW("{}: string table is missing or malformed", Path);
  StrTab = getContents(Shdrs[hdr.e_shstrndx]);

  for (auto const &shdr : Shdrs)
    if ((shdr.sh_offset | shdr.sh_addr) & 3
        || shdr.sh_offset + shdr.sh_size > Owner.Contents.size())
      TT_THROW("{}: section {} is misaligned", Path, getName(shdr));

  bool foundText = false;
  for (auto const &phdr : Phdrs) {
    if (phdr.p_type == PT_RISCV_ATTRIBUTES) {
      Arch = getArchFromAttrs(getContents(phdr));
      // FIXME: verify Arch is ok?
      continue;
    }
    if (phdr.p_type != PT_LOAD)
      continue;
    if (!phdr.p_memsz)
      // Have observed zero-sized segments, ignore them
      continue;

    if ((phdr.p_offset | phdr.p_vaddr | phdr.p_filesz | phdr.p_memsz) & 3)
      TT_THROW("{}: loadable segment {} is misaligned", Path,
               unsigned(Owner.Segments.size()));

    auto contents = getContents(phdr);
    if ((phdr.p_flags & PF_X) && !(phdr.p_flags & PF_W)) {
      // text
      if (foundText)
        TT_THROW("{}: multiple text segments found", Path);
      foundText = true;
      address_t entry = getHeader().e_entry - phdr.p_vaddr;
      if (entry >= phdr.p_memsz)
        TT_THROW("{}: entry point is not in text segment", Path);
      Owner.Segments.insert(
          Owner.Segments.begin(),
          Segment(contents, phdr.p_vaddr, entry));
    } else
      Owner.Segments.emplace_back(contents, phdr.p_vaddr,
				  phdr.p_memsz - phdr.p_filesz);
  }
  if (!foundText)
    TT_THROW("{}: cannot find text segment", Path);
}

void ElfFile::Impl::relocateImage () {
  auto isInSegment
      = [&] (Segment const &segment, Elf32_Shdr const &shdr) -> bool {
    return shdr.sh_addr >= segment.Address
           && (shdr.sh_addr + shdr.sh_size <= segment.Address
	       + segment.Contents.size() + segment.EntryOrBss);
  };
  // Is SHDR in the text segment?
  auto isText = [&] (Elf32_Shdr const &shdr) -> bool {
    return isInSegment(Owner.Segments.front(), shdr);
  };
  auto getSegmentIx = [&] (Elf32_Shdr const &shdr) -> int {
    for (unsigned ix = Owner.Segments.size(); ix--;)
      if (isInSegment(Owner.Segments[ix], shdr))
        return ix;
    return -1;
  };
  auto checkReloc = [&] (Elf32_Rela const &reloc, Elf32_Shdr const &section) {
    bool outside = reloc.r_offset - section.sh_addr >= section.sh_size;
    if (outside || reloc.r_offset & 3)
      TT_THROW("{}: relocation @ {x} is {} section {}", Path, reloc.r_offset,
               outside ? "outside of" : "misaligned in", getName(section));
  };
  auto checkRelocPair = [&] (bool isPcrel, bool loIsFormI,
                             Elf32_Rela const &loReloc,
                             Elf32_Rela const *hiReloc, uint32_t relaxOffset,
                             uint32_t hiRelaxOffset) {
    if (!hiReloc)
      TT_THROW("{}: R_RISCV{}_LO12_{} relocation at {x} has"
               " no matching R_RISCV{}_HI20",
               Path, isPcrel ? "_PCREL" : "", loIsFormI ? 'I' : 'S',
               loReloc.r_offset, isPcrel ? "_PCREL" : "");
    if (loReloc.r_offset != relaxOffset)
      log_debug(tt::LogLLRuntime,
                "{}: R_RISCV{}_LO12_{} relocation at {x} is not relaxed", Path,
                isPcrel ? "_PCREL" : "", loIsFormI ? 'I' : 'S',
                loReloc.r_offset);
    if (hiReloc->r_offset != hiRelaxOffset)
      log_debug(tt::LogLLRuntime,
                "{}: R_RISCV{}_HI20 relocation at {x} is not relaxed", Path,
                isPcrel ? "_PCREL" : "", hiReloc->r_offset);
    log_debug(tt::LogLLRuntime,
              "{}: translating R_RISCV{}_{HI20/LO12_{}} pair at {x} & {x}"
              " to R_RISCV{}_[HI20/LO12_{}} pair",
              Path, isPcrel ? "_PCREL" : "", loIsFormI ? 'I' : 'S',
              hiReloc->r_offset, loReloc.r_offset, !isPcrel ? "_PCREL" : "",
              loIsFormI ? 'I' : 'S');
  };
  auto translateInsnPair = [&] (std::span<std::byte> contents,
                                unsigned offsetHi, unsigned offsetLo,
                                uint32_t insnHiOpc, bool loIsFormI,
                                uint32_t value) {
    uint32_t hiInsn = read32(contents, offsetHi);
    if ((hiInsn & insn_mask_u) != insnHiOpc)
      TT_THROW("{}: translating instruction at {x} is not `{}'", Path,
               offsetHi, insnHiOpc == insn_opc_auipc ? "auipc" : "lui");
    hiInsn &= mask_hi20;                     // Remove old immediate
    hiInsn ^= insn_opc_auipc ^ insn_opc_lui; // Convert opcode
    // Insert new immediate
    hiInsn |= ((value + (1 << 11)) >> 12) << mask_hi20_shift;
    write32(contents, offsetHi, hiInsn);

    uint32_t loInsn = read32(contents, offsetLo);
    if (loIsFormI) {
      loInsn &= mask_lo12_i;
      loInsn |= (value & 0x0fff) << mask_lo12_i_shift;
    } else {
      // S form splits the immediate
      loInsn &= mask_lo12_s;
      loInsn |= (value & ((1 << mask_lo12_s_split) - 1))
                << mask_lo12_s_shift_1;
      loInsn |= ((value & 0x0fff) >> mask_lo12_s_split) << mask_lo12_s_shift_2;
    }
    write32(contents, offsetLo, loInsn);
  };

  unsigned count = 0;
  for (auto const &relocHdr : Shdrs) {
    if (relocHdr.sh_type != SHT_RELA)
      continue;

    count++;
    // Is this relocating a section of interest?
    unsigned sectionIx = relocHdr.sh_info;
    auto &section = Shdrs[sectionIx];
    if (section.sh_type != SHT_PROGBITS && section.sh_type != SHT_INIT_ARRAY
        && section.sh_type != SHT_FINI_ARRAY
        && section.sh_type != SHT_PREINIT_ARRAY)
      continue;

    int segmentIx = getSegmentIx(section);
    if (segmentIx < 0)
      continue;

    auto symbols = getSymbols(Shdrs[relocHdr.sh_link]);
    auto contents = getContents(section);
    auto relocs = getRelocations(relocHdr);
    bool fromText = !segmentIx;

    // The RELAX relocation appears immediately after the relocation
    // that can be relaxed.  Thus it's easier to scan in reverse.
    uint32_t relaxOffset = ~0u;
    for (unsigned ix = relocs.size(); ix--;) {
      auto &reloc = relocs[ix];

      checkReloc(reloc, section);

      auto type = ELF32_R_TYPE(reloc.r_info);
      auto symIx = ELF32_R_SYM(reloc.r_info);
      auto const *symbol = &symbols[symIx];
      bool toText = (symbol->st_shndx < Shdrs.size())
                    && isText(Shdrs[symbol->st_shndx]);
      bool loIsFormI = false;

      switch (type) {
      case R_RISCV_LO12_I:
        loIsFormI = true;
        [[fallthrough]];
      case R_RISCV_LO12_S: {
        if (!(fromText && toText))
          break;

        // Scan up to find the matchin HI reloc
        Elf32_Rela *hiReloc = nullptr;
        uint32_t hiRelaxOffset = relaxOffset;
        for (unsigned probeIx = ix; probeIx--;) {
          auto &probe = relocs[probeIx];
          if (ELF32_R_TYPE(probe.r_info) == R_RISCV_RELAX)
            hiRelaxOffset = probe.r_offset;
          else if (ELF32_R_TYPE(probe.r_info) == R_RISCV_HI20
                   && ELF32_R_SYM(probe.r_info) == symIx
                   && probe.r_addend == reloc.r_addend) {
            hiReloc = &probe;
            break;
          }
        }

        checkRelocPair(false, loIsFormI, reloc, hiReloc, relaxOffset,
                       hiRelaxOffset);
        checkReloc(*hiReloc, section);

        // Convert to R_RISCV_PCREL_{HI20, LO12_{IS]} pair
        translateInsnPair(
            contents, hiReloc->r_offset - section.sh_addr,
            reloc.r_offset - section.sh_addr, insn_opc_lui, loIsFormI,
            symbol->st_value + hiReloc->r_addend - hiReloc->r_offset);

        // We can't convert with fidelity, as that involves adding a
        // symbol. Instead, let's true a null symbol and an addend.
        hiReloc->r_info = ELF32_R_INFO(symIx, R_RISCV_PCREL_HI20);
        reloc.r_info = ELF32_R_INFO(0, loIsFormI ? R_RISCV_PCREL_LO12_I
                                                 : R_RISCV_PCREL_LO12_S);
        reloc.r_addend = hiReloc->r_offset - reloc.r_offset;

        break;
      }

      case R_RISCV_HI20:
        // handled above.
        break;

      case R_RISCV_PCREL_LO12_I:
        loIsFormI = true;
        [[fallthrough]];
      case R_RISCV_PCREL_LO12_S: {
        // Scan up to find the matchin HI reloc.  The SYMBOL we have
        // points at that reloc, the hi reloc has the symbol we're
        // relocating.
        Elf32_Rela *hiReloc = nullptr;
        uint32_t hiRelaxOffset = relaxOffset;
        uint32_t hiOffset = symbol->st_value + reloc.r_addend;
        for (unsigned probeIx = ix; probeIx--;) {
          auto &probe = relocs[probeIx];
          if (ELF32_R_TYPE(probe.r_info) == R_RISCV_RELAX)
            hiRelaxOffset = probe.r_offset;
          else if (ELF32_R_TYPE(probe.r_info) == R_RISCV_PCREL_HI20
                   && probe.r_offset != hiOffset) {
            // This reloc holds the symbol of interest. Recompute some things.
            symIx = ELF32_R_SYM(probe.r_info);
            symbol = &symbols[symIx];
            toText = (symbol->st_shndx < Shdrs.size()
                      && isText(Shdrs[symbol->st_shndx]));
            hiReloc = &probe;
            break;
          }
        }

        // We only know toText if hiReloc is nonnull. If it's null
        // we'll barf in checkRelocPair.
        if (hiReloc && fromText == toText)
          break;

        checkRelocPair(true, loIsFormI, reloc, hiReloc, relaxOffset,
                       hiRelaxOffset);
        checkReloc(*hiReloc, section);

        // Convert to R_RISCV_{HI20,LO12_[IS]} pair
        translateInsnPair(contents, hiReloc->r_offset - section.sh_addr,
                          reloc.r_offset - section.sh_addr, insn_opc_auipc,
                          loIsFormI, symbol->st_value + hiReloc->r_addend);

        hiReloc->r_info = ELF32_R_INFO(symIx, R_RISCV_HI20);
        reloc.r_info
            = ELF32_R_INFO(symIx, loIsFormI ? R_RISCV_LO12_I : R_RISCV_LO12_S);
        reloc.r_addend = hiReloc->r_addend;
        break;
      }

      case R_RISCV_PCREL_HI20:
        // Handled above.
        break;

      case R_RISCV_32: {
        if (!toText)
          break;
        // Emit dynamic reloc
        log_debug(tt::LogLLRuntime,
                  "{}: emitting dynamic R_RISCV_32 relocation at {x}", Path,
                  reloc.r_offset);
        address_t value = (symbol->st_value + reloc.r_addend
                           - Owner.Segments.front().Address);
        write32(getContents(section), reloc.r_offset - section.sh_addr, value);
	auto &seg = Owner.Segments[segmentIx];
        seg.Relocs.push_back (reloc.r_offset - seg.Address);
      } break;

      case R_RISCV_JAL:
        if (fromText != toText)
          TT_THROW("{}: segment-crossing R_RISCV_JAL relocation found at {x}",
                   Path, reloc.r_offset);
        break;

      case R_RISCV_CALL:
      case R_RISCV_CALL_PLT:
        TT_THROW("{}: R_RISCV_CALL{,_PLT} relocation found at {x}", Path,
                 reloc.r_offset);
        break;

      case R_RISCV_32_PCREL:
        TT_THROW("{}: R_RISCV_32_PCREL relocation found at {x}", Path,
                 reloc.r_offset);
        break;

      case R_RISCV_RELAX:
        relaxOffset = reloc.r_offset;
        break;
      }
    }
  }

  if (!count)
    // Hm, that's suspicious
    TT_THROW("{}: there are no relocations", Path);

  // The text segment is now XIP
  Owner.Segments.front().Address = 0;
}

char const *ElfFile::Impl::getArchFromAttrs (std::span<std::byte> attribs) {
  // Attributes are <key, value> tuples. Even keys have a uleb128
  // value. Odd keys have a nul-terminated string. These are
  // gnu_attributes form.
  // char: version - 'A'
  // uint32: attr-len, including these bytes
  // NTBS: attr-name - "riscv"
  // { repeat
  // uint8: tag - 0x1
  // uint32: len - including these bytes
  // uleb128: tag
  // ntbs : name (odd tag)
  // uleb128: value (even tag)
  // }

  // FIXME: Implement
  // Contents of section .riscv.attributes:
  //  0000 41200000 00726973 63760001 16000000  A ...riscv......
  //  0010 04100572 76333269 3270305f 6d327030  ...rv32i2p0_m2p0
  //  0020 00
  return nullptr;
}
