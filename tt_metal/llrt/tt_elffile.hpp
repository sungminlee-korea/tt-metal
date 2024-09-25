// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// C++
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

// An ELF executable loader
// This is a replacement for tt_hexfile stuff.

namespace ll_api {

class ElfFile {
public:
  // ELF32
  using address_t = std::uint32_t; // Address in memory
  using offset_t = std::uint32_t;  // Offset within region
  using word_t = std::uint32_t;    // Contents

  struct Segment {
    std::span<word_t const> Contents; // Non-owning span
    address_t Address = 0;   // word addres or 0 for XIP
    offset_t EntryOrBSS = 0; // word text entry or data bss

  public:
    constexpr Segment (std::span<word_t const> contents, address_t addr,
		       offset_t entryOrBSS)
      : Contents(contents), Address(addr), EntryOrBSS(entryOrBSS) {}
  };

public:
  ElfFile (std::string const &path);
  ~ElfFile ();

public:
  std::vector<Segment> const &getSegments () const { return Segments; }

private:
  class Impl;
  std::span<std::byte> Contents; // Owning buffer
  std::vector<Segment> Segments;
};

} // namespace ll_api
