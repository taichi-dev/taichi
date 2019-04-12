#include <iostream>
#include <taichi/testing.h>
#include "../tlang.h"
#include <taichi/common/util.h>
#include <taichi/math.h>
#include <taichi/common/testing.h>
#include <taichi/system/timer.h>
#include <unordered_map>

using namespace taichi;
using Tlang::measure_cpe;

struct Node {
  float x, y;
};
struct Block {
  static constexpr int size = 256;
  Node nodes[size];
};
struct Tile {
  static constexpr int size = 1024;
  Block *blocks[size];
};
constexpr int dim1 = Block::size;
constexpr int dim0 = Tile::size * dim1;
std::unordered_map<int, Tile> data;

inline float safe_access_x(int i) {
  auto it = data.find(i / dim0);
  if (it == data.end())
    return 0;
  auto b = it->second.blocks[i % dim0 / dim1];
  if (b == nullptr)
    return 0;
  return b->nodes[i % dim1].x;
}

void copy_ref() {
  for (auto &it : data) {
    auto &tile = it.second;
    for (int b = 0; b < Tile::size; b++) {
      auto block = tile.blocks[b];
      if (!block)
        continue;
      for (int n = 0; n < Block::size; n++) {
        int i = it.first * dim0 + b * dim1 + n;
        block->nodes[n].y = safe_access_x(i);
      }
    }
  }
}

void copy_optimized() {
  for (auto &it : data) {
    auto &tile = it.second;
    for (int b = 0; b < Tile::size; b++) {
      auto block = tile.blocks[b];
      if (!block)
        continue;
      for (int n = 0; n < Block::size; n++) {
        block->nodes[n].y = block->nodes[n].x;
      }
    }
  }
}

void stencil_ref() {
  for (auto &it : data) {
    auto &tile = it.second;
    for (int b = 0; b < Tile::size; b++) {
      auto block = tile.blocks[b];
      if (!block)
        continue;
      for (int n = 0; n < Block::size; n++) {
        int i = it.first * dim0 + b * dim1 + n;
        block->nodes[n].y =
            (1.0f / 3) *
            (safe_access_x(i - 1) + safe_access_x(i) /* Note: can be weakened*/
             + safe_access_x(i + 1));
      }
    }
  }
}

void stencil_optimized() {
  for (auto &it : data) {
    auto &tile = it.second;
    for (int b = 0; b < Tile::size; b++) {
      auto block = tile.blocks[b];
      if (!block)
        continue;
      for (int n = 0; n < Block::size; n++) {
        int i = it.first * dim0 + b * dim1 + n;
        auto tmp = block->nodes[n].x;
        if (n > 0) {
          tmp += block->nodes[n - 1].x;
        } else {
          tmp += safe_access_x(i - 1);
        }
        if (n < Block::size - 1) {
          tmp += block->nodes[n + 1].x;
        } else {
          tmp += safe_access_x(i + 1);
        }
        block->nodes[n].y = (1.0f / 3) * tmp;
      }
    }
  }
}

void stencil_optimized2() {
  for (auto &it : data) {
    auto &tile = it.second;
    for (int b = 0; b < Tile::size; b++) {
      auto block = tile.blocks[b];
      if (!block)
        continue;
      block->nodes[0].y =
          (1.0f / 3) * (safe_access_x(it.first * dim0 + b * dim1 - 1) +
                        block->nodes[0].x + block->nodes[1].x);
      for (int n = 1; n < Block::size - 1; n++) {
        auto tmp = block->nodes[n].x;
        tmp += block->nodes[n - 1].x;
        tmp += block->nodes[n + 1].x;
        block->nodes[n].y = (1.0f / 3) * tmp;
      }
      block->nodes[Block::size - 1].y =
          (1.0f / 3) *
          (block->nodes[Block::size - 2].x + block->nodes[Block::size - 1].x +
           safe_access_x(it.first * dim0 + b * dim1 + Block::size));
    }
  }
}

void stencil_optimized3() {
  Block dummy;
  std::memset(&dummy, 0, sizeof(dummy));
  for (auto &it : data) {
    auto &tile = it.second;
    for (int b = 0; b < Tile::size; b++) {
      auto block = tile.blocks[b];
      if (!block)
        continue;
      Block *blocks[3];
      blocks[1] = block;
      blocks[0] = nullptr;
      blocks[2] = nullptr;
      if (b > 0)
        blocks[0] = tile.blocks[b - 1];
      else {
        auto prev = data.find(it.first - 1);
        if (prev != data.end())
          blocks[0] = prev->second.blocks[Tile::size - 1];
      }
      if (b + 1 < Tile::size)
        blocks[2] = tile.blocks[b + 1];
      else {
        auto next = data.find(it.first + 1);
        if (next != data.end())
          blocks[2] = next->second.blocks[0];
      }
      if (blocks[0] == nullptr) {
        blocks[0] = &dummy;
      }

      if (blocks[2] == nullptr) {
        blocks[2] = &dummy;
      }

      for (int n = 0; n < Block::size; n++) {
        auto tmp = block->nodes[n].x;
        tmp += blocks[(n - 1 + Block::size) / Block::size]
                   ->nodes[(n - 1 + Block::size) % Block::size]
                   .x;
        tmp += blocks[(n + 1 + Block::size) / Block::size]
                   ->nodes[(n + 1) % Block::size]
                   .x;
        block->nodes[n].y = (1.0f / 3) * tmp;
      }
    }
  }
}

void benchmark_layers() {
  int n = 1000000;
  int cnt = 0;
  int cnt2;

  int t = 0;

  auto test_hash_table = [&]() {
    for (int i = 0; i < n; i++) {
      t = (t + 7) & 1023;
      if (data.find(t) != data.end()) {
        cnt += 1;
      }
    }
  };

  TC_P(measure_cpe(test_hash_table, n));

  auto &tile = data.begin()->second;

  auto test_block = [&]() {
    for (int i = 0; i < n; i++) {
      t = (t + 7) & Tile::size;
      if (tile.blocks[t] != nullptr) {
        cnt += ((int64)tile.blocks[t] & 31);
      }
    }
  };

  TC_P(measure_cpe(test_block, n));

  TC_P(cnt);
}

TLANG_NAMESPACE_BEGIN

TC_TEST("stencil1d") {
  CoreState::set_trigger_gdb_when_crash(true);
  Program prog;

  AmbientGlobal(x, f32, 0.0f);
  Global(y, f32);

  layout([&] {
    auto i = Index(0);
    root.hashed(i, 8192).dense(i, 1024).pointer().dense(i, 256).place(x, y);
  });

  auto &copy = kernel([&] {
    Declare(i);
    // Cache(0);
    For(i, x, [&] { y[i] = x[i]; });
  });

  auto &copy_parallelized = kernel([&] {
    Declare(i);
    Parallelize(4);
    // Cache(0);
    For(i, x, [&] { y[i] = x[i]; });
  });

  auto &stencil = kernel([&] {
    Declare(i);
    // Cache(0);
    For(i, x, [&] { y[i] = (1.0f / 3) * (x[i - 1] + x[i] + x[i + 1]); });
  });

  int total_nodes = 0;
  {
    // initialize
    int total_tiles = 0;
    int total_blocks = 0;
    for (int i = 0; i < 8192; i++) {
      if (i % 31 != 5) {
        continue;
      }
      total_tiles++;
      auto &tile = data[i];
      std::memset(&tile, 0, sizeof(tile));
      for (int j = 0; j < Tile::size; j++) {
        auto key = j % 37;
        if (!(12 <= key && key < 14)) {
          continue;
        }
        auto b = new ::Block;
        total_blocks++;
        tile.blocks[j] = b;
        for (int k = 0; k < ::Block::size; k++) {
          auto val = taichi::rand();
          b->nodes[k].x = val;
          b->nodes[k].y = 0;
          total_nodes += 1;
          auto index = i * dim0 + j * dim1 + k;
          x.val<float32>(index) = val;
        }
      }
    }
    TC_P(total_tiles);
    TC_P(total_blocks);
    TC_P(total_nodes);
  }
  // benchmark_layers();
  // TC_P(measure_cpe(stencil, total_nodes));

  for (int i = 0; i < 10; i++)
    TC_TIME(copy_ref());

  for (int i = 0; i < 10; i++)
    TC_TIME(copy_optimized());

  for (int i = 0; i < 10; i++)
    TC_TIME(copy());

  for (int i = 0; i < 10; i++)
    TC_TIME(copy_parallelized());

  // test copy to x
  for (auto &it : data) {
    auto &tile = it.second;
    for (int b = 0; b < Tile::size; b++) {
      auto block = tile.blocks[b];
      if (!block)
        continue;
      for (int n = 0; n < ::Block::size; n++) {
        int i = it.first * dim0 + b * dim1 + n;
        TC_CHECK(block->nodes[n].y == y.val<float32>(i));
      }
    }
  }

  for (int i = 0; i < 10; i++)
    TC_TIME(stencil_ref());

  for (int i = 0; i < 10; i++)
    TC_TIME(stencil_optimized());

  for (int i = 0; i < 10; i++)
    TC_TIME(stencil_optimized2());

  for (int i = 0; i < 10; i++)
    TC_TIME(stencil_optimized3());

  for (int i = 0; i < 10; i++)
    TC_TIME(stencil());

  // test stencil to x
  for (auto &it : data) {
    auto &tile = it.second;
    for (int b = 0; b < Tile::size; b++) {
      auto block = tile.blocks[b];
      if (!block)
        continue;
      for (int n = 0; n < ::Block::size; n++) {
        int i = it.first * dim0 + b * dim1 + n;
        TC_CHECK_EQUAL(block->nodes[n].y, y.val<float32>(i), 1e-5f);
      }
    }
  }
}

TLANG_NAMESPACE_END
