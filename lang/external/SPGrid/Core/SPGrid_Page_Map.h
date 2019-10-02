//#####################################################################
// Copyright (c) 2012-2016, Eftychios Sifakis, Sean Bauer
// Distributed under the FreeBSD license (see license.txt)
//#####################################################################
#ifndef __SPGrid_Page_Map_h__
#define __SPGrid_Page_Map_h__

#include <vector>
#include <mutex>

namespace SPGrid {
//#####################################################################
// Class SPGrid_Page_Map
//#####################################################################
template <int log2_page = 12>
class SPGrid_Page_Map {
 protected:
  const uint64_t map_size;  // Size of the page map, in uint64_t units. Each
                            // entry corresponds to 64 (1<<log2_page) pages.
  uint64_t *page_map;  // The actual page map - a bitmap structured as an array
                       // of 64-bit entries.
  std::vector<uint64_t> block_offsets;  // Alternative representation as a list
                                        // of linearized offsets. Created on
                                        // demand.
  bool dirty;  // Indicates that block offsets are inconsistent with the page
               // map (perhaps for a good reason, if only one of them is used).
  std::mutex mutex;

 public:
  // Make the pagemap class noncopyable
  SPGrid_Page_Map(const SPGrid_Page_Map &) = delete;
  SPGrid_Page_Map &operator=(const SPGrid_Page_Map &) = delete;

  template <int d>
  SPGrid_Page_Map(const SPGrid_Geometry<d> &geometry)
      : map_size((geometry.Padded_Volume() / geometry.Elements_Per_Block() +
                  0x3fUL) >>
                 6) {
    page_map =
        static_cast<uint64_t *>(Raw_Allocate(map_size * sizeof(uint64_t)));
    dirty = false;
  }

  ~SPGrid_Page_Map() {
    Raw_Deallocate(page_map, map_size * sizeof(uint64_t));
  }

  void Clear_Page_Map() {
    Raw_Deallocate(page_map, map_size * sizeof(uint64_t));
    page_map =
        static_cast<uint64_t *>(Raw_Allocate(map_size * sizeof(uint64_t)));
    dirty = true;
  }

  void Clear_Blocks() {
    std::vector<uint64_t>().swap(block_offsets);
    dirty = true;
  }

  void Clear() {
    Clear_Page_Map();
    Clear_Blocks();
    dirty = false;
  }

  void Set_Page(const uint64_t offset) {
    uint64_t mask = 1UL << (offset >> log2_page & 0x3f);
    uint64_t &entry = page_map[offset >> (log2_page + 6)];
    if (mask & ~entry) {
      mutex.lock();
      entry |= mask;
      mutex.unlock();
    }
    if (!dirty)
      dirty = true;  // Important to avoid unnecessary write sharing
  }

  void Unset_Page(const uint64_t offset) {
    uint64_t mask = 1UL << (offset >> log2_page & 0x3f);
    uint64_t &entry = page_map[offset >> (log2_page + 6)];
    if (mask & entry) {
      mutex.lock();
      entry ^= mask;
      mutex.unlock();
    }
    if (!dirty)
      dirty = true;  // Important to avoid unnecessary write sharing
  }

  bool Test_Page(const uint64_t offset) const {
    uint64_t mask = 1UL << (offset >> log2_page & 0x3f);
    if (offset >> (log2_page + 6) >= map_size)
      return false;
    const uint64_t &entry = page_map[offset >> (log2_page + 6)];
    return entry & mask;
  }

  std::pair<const uint64_t *, unsigned> Get_Blocks() const {
    if (block_offsets.size())
      return std::pair<const uint64_t *, unsigned>(&block_offsets[0],
                                                   block_offsets.size());
    else
      return std::pair<const uint64_t *, unsigned>((const uint64_t *)0, 0);
  }

  void Update_Block_Offsets() {
    std::vector<uint64_t> new_block_offsets(Generate_Block_Offsets());
    if (dirty)
      block_offsets.swap(new_block_offsets);
    dirty = false;
  }

  // This implementation is currently suboptimal in that it will touch the
  // entirety of the page map.
  // It should perferably be implemented using mincore() instead, to selectively
  // query only resident pages.
  std::vector<uint64_t> Generate_Block_Offsets() {
    std::vector<uint64_t> block_offsets;
    for (uint64_t entry = 0; entry < map_size; entry++)
      if (page_map[entry])
        for (uint64_t pos = 0; pos < 64; pos++)
          if (page_map[entry] & (1UL << pos))
            block_offsets.push_back((entry << (log2_page + 6)) |
                                    (pos << log2_page));
    return block_offsets;
  }

  //#####################################################################
};
}
#endif
