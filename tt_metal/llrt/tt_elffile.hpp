// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// C++
#include <cstddef>
//#include <memory>
#include <span>
#include <vector>

// An ELF executable loader
// This is a replacement for tt_hexfile stuff.

namespace ll_api {

class ElfFile {
public:
  // ELF32
  using address_t = std::uint32_t;
  using offset_t = std::uint32_t;

  struct Segment {
    std::span<std::byte const> Contents; // Non-owning span
    address_t Address = 0;               // vaddr or 0 for XIP
    offset_t EntryOrBss = 0; // text entry or data bss

  public:
    constexpr Segment (std::span<std::byte const> contents, address_t addr,
		       offset_t bssOrEntry)
      : Contents(contents), Address(addr), EntryOrBss(bssOrEntry) {}
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
