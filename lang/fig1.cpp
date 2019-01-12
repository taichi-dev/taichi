#include <iostream>
#include "util.h"
#include "tlang.h"
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
  if (data.find(i / dim0) == data.end())
    return 0;
  if (data[i / dim0].blocks[i % dim0 / dim1] == nullptr)
    return 0;
  return data[i / dim0].blocks[i % dim0 / dim1]->nodes[i % dim1].x;
}

void stencil() {
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

  return;
  Program prog;

  auto x = var<float32>(), y = var<float32>(), i = ind();
  layout([&] {
    root.hashed(i, 1024).fixed(i, 1024).pointer()
        .fixed(i, 256).place(x, y);
  });
  auto stencil = kernel(x, [&] {
    //x[i] = imm(1.0f / 3) * (y[i - imm(1)] + y[i] + y[i + imm(1)]);
    x[i] = load(y[i]);
    // y[i] = imm(0);//y[i];
    // x[i] = y[i];
  });

  int total_nodes = 0;
  {
    // initialize
    int total_tiles = 0;
    int total_blocks = 0;
    for (int i = 0; i < 1024; i++) {
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
        auto b = new Block;
        total_blocks++;
        tile.blocks[j] = b;
        for (int k = 0; k < Block::size; k++) {
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

  TC_TIME(stencil());
}

TLANG_NAMESPACE_END
