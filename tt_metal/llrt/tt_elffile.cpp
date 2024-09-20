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
