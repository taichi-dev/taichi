#include "taichi_grid.h"
#include <taichi/system/threading.h>

TC_NAMESPACE_BEGIN

using Block = TestGrid::Block;

MPIEnvironment mpi_env;

TC_TEST("dilated block") {
  if (with_mpi())
    return;
  using Block = TBlock<int, char, TSize3D<8>, 1>;
  Block block(Vector3i(8));

  TC_STATIC_ASSERT(Block::num_nodes == pow<3>(10));
  CHECK(block.linearize_global(Vector3i(7)) == 0);

  int n = 8;
  for (int i = -1; i <= n; i++) {
    for (int j = -1; j <= n; j++) {
      for (int k = -1; k <= n; k++) {
        block.node_global(Vector3i(8) + Vector3i(i, j, k)) = i * j + k;
      }
    }
  }
  for (int i = -1; i <= n; i++) {
    for (int j = -1; j <= n; j++) {
      for (int k = -1; k <= n; k++) {
        CHECK(block.node_global(Vector3i(8) + Vector3i(i, j, k)) == i * j + k);
      }
    }
  }

  using Grid = TaichiGrid<Block>;
  Grid grid;
  Vector3i block_size(8);
  Region3D block_region(Vector3i(7), Vector3i(10));
  Region3D local_grid_region(Vector3i(-1), Vector3i(1) + block_size);

  TArray<3, int> gt(Vector3i(100));
  for (auto b_ind : block_region) {
    auto base_coord = b_ind.get_ipos() * block_size;
    grid.touch(base_coord);
    auto b = grid.get_block_if_exist(base_coord);
    for (auto i : local_grid_region) {
      auto val = rand_int();
      b->node_local(i.get_ipos()) = val;
      gt[base_coord + i.get_ipos()] += val;
    }
  }

  // Do grid exchange here
  grid.advance([](Block &b, TAncestors<Block> &an) {
    accumulate_dilated_grids(b, an);
    return true;
  });

  for (auto b_ind : block_region) {
    auto base_coord = b_ind.get_ipos() * block_size;
    auto b = grid.get_block_if_exist(base_coord);
    for (auto i : local_grid_region) {
      CHECK(gt[base_coord + i.get_ipos()] == b->node_local(i.get_ipos()));
    }
  }
}

TC_TEST("grid_basics") {
  CHECK(product<int, 3>(std::array<int, 3>({2, 3, 4})) == 24);
  CHECK(product<int, 1>(std::array<int, 1>({7})) == 7);

  CHECK(least_pot_bound(7) == 8);
  CHECK(least_pot_bound(0) == 1);
  CHECK(least_pot_bound(1) == 1);
  CHECK(least_pot_bound(1024) == 1024);
  CHECK(least_pot_bound(1023) == 1024);
  CHECK(least_pot_bound(1025) == 2048);

  CHECK(pdep(7, 7) == 7);
  CHECK(pdep(7, 14) == 14);
  CHECK(pdep(3, 14) == 6);
  CHECK(pdep(3, 0) == 0);
  CHECK(pdep(0, 3) == 0);
  CHECK(pdep(1, 3) == 1);
  CHECK(pdep(2, 3) == 2);
  CHECK(pdep(3, 3) == 3);
  CHECK(pdep(1, 21) == 1);
  CHECK(pdep(2, 21) == 4);
  CHECK(pdep(3, 21) == 5);
  CHECK(pdep(4, 21) == 16);

  CHECK(log2int(4) == 2);
  CHECK(log2int(1) == 0);
  CHECK(log2int(8) == 3);
  CHECK(log2int(1ll << 50) == 50);

  CHECK(pot_mask(8) == 255);
}

TC_TEST("grid") {
  if (with_mpi())
    return;
  TestGrid grid;

  constexpr int n = 136;

  CHECK(div_floor(Vector3i(-1, -7, -8), Vector3i(8)) == Vector3i(-1, -1, -1));

  TC_STATIC_ASSERT(n % TestGrid::Block::size[0] == 0);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        auto coord = Vector3i(i, j, k);
        grid.touch(coord);
        grid.node(coord).x = i + j * k;
        CHECK(grid.node(coord).x == i + j * k);
      }
    }
  }
  CHECK(grid.root.size() == pow<3>((n - 1) / 128 + 1));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        CHECK(grid.node(Vector3i(i, j, k)).x == i + j * k);
      }
    }
  }
  grid.for_each_block([&](Block &b) {
    for (int i = 0; i < b.num_nodes; i++) {
      b.nodes[i].x += 1;
    }
  });
  grid.for_each_node([&](Block::Node &n) { n.x *= 2; });
  int64 sum = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        auto coord = Vector3i(i, j, k);
        CHECK(grid.node(coord).x == (i + j * k + 1) * 2);
        sum += (int64)(grid.node(Vector3i(i, j, k)).x);
      }
    }
  }
  auto func = [](Block &b) -> int64 {
    int64 sum = 0;
    for (auto n : b.nodes) {
      sum += int64(n.x);
    }
    return sum;
  };
  CHECK(grid.reduce(func, std::plus<int64>(), 0) == sum);
  CHECK(grid.reduce(func, std::plus<int64>()) == sum);
  CHECK(grid.reduce(func) == sum);
  grid.for_each_block([&](Block &b) {
    for (int i = 0; i < Block::size[0]; i++) {
      for (int j = 0; j < Block::size[1]; j++) {
        for (int k = 0; k < Block::size[2]; k++) {
          b.node_local(Vector3i(i, j, k)) =
              (b.base_coord + Vector3i(i, j, k)).template cast<real>();
        }
      }
    }
  });
  grid.advance([&](Block &b, TestGrid::Ancestors &an) {
    auto scratch = TestGrid::GridScratchPad(an);
    auto base_coord = b.base_coord;
    int p = 0;
    for (int i = -1; i <= Block::size[0]; i++) {
      for (int j = -1; j <= Block::size[1]; j++) {
        for (int k = -1; k <= Block::size[2]; k++) {
          auto a = scratch.linearized_data[p];
          auto coord = base_coord + Vector3i(i, j, k);
          if (Vector3i(0) <= coord && coord < Vector3i(n)) {
            auto b = coord.cast<real>();
            CHECK(a == b);
          }
          p++;
        }
      }
    }
    return true;
  });
}

TC_TEST("block base") {
  {
    // Test at coord 0
    Block base(Vector3i(0));
    int n = 8;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
          base.node_global(Vector3i(i, j, k)).x = i + j * k;
        }
      }
    }
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
          CHECK(base.node_global(Vector3i(i, j, k)).x == i + j * k);
        }
      }
    }
  }
}

TC_TEST("Propagate") {
  if (with_mpi()) {
    return;
  }
  TestGrid grid;
  grid.touch(Vector3i(0));
  grid.node(Vector3i(0)).x = 100;
  for (int i = 0; i < 10; i++) {
    if (i == 0)
      CHECK(grid.num_active_blocks() == 1);
    if (i == 1)
      CHECK(grid.num_active_blocks() == 4);
    if (i == 2)
      CHECK(grid.num_active_blocks() == 7);
    grid.advance([&](Block &b, TestGrid::Ancestors &an) {
      auto scratch = TestGrid::GridScratchPad(an);
      bool has_non_zero = false;
      for (int i = 0; i < Block::size[0]; i++) {
        for (int j = 0; j < Block::size[1]; j++) {
          for (int k = 0; k < Block::size[2]; k++) {
            int maximum = 0;
            auto update = [&](int di, int dj, int dk) {
              maximum = std::max(maximum,
                                 int(scratch.data[i + di][j + dj][k + dk].x));
            };
            update(0, 0, 0);
            update(1, 0, 0);
            update(-1, 0, 0);
            update(0, 1, 0);
            update(0, -1, 0);
            update(0, 0, 1);
            update(0, 0, -1);
            if (maximum != 0) {
              b.node_local(Vector3i(i, j, k)).x = maximum;
              has_non_zero = true;
            }
          }
        }
      }
      return has_non_zero;
    });
    // TC_P(grid.num_active_blocks());
  }
  CHECK(int(grid.node(Vector3i(0, 10, 0)).x) == 100);
  CHECK(int(grid.node(Vector3i(0, 11, 0)).x) == 0);
  CHECK(int(grid.node(Vector3i(10, 0, 0)).x) == 100);
  CHECK(int(grid.node(Vector3i(11, 0, 0)).x) == 0);
  CHECK(int(grid.node(Vector3i(-10, 0, 0)).x) == 100);
  CHECK(int(grid.node(Vector3i(-11, 0, 0)).x) == 0);
  CHECK(int(grid.node(Vector3i(0, 0, 10)).x) == 100);
  CHECK(int(grid.node(Vector3i(0, 0, 11)).x) == 0);

  CHECK(int(grid.node(Vector3i(0, 5, 5)).x) == 100);
  CHECK(int(grid.node(Vector3i(0, 6, 5)).x) == 0);

  CHECK(grid.root.size() == 8);
}

TC_TEST("basic distributed 2") {
  if (!with_mpi())
    return;
  TestGrid grid;
  if (grid.world_size != 2) {
    return;
  }
  if (grid.world_rank == 0) {
    grid.touch(Vector3i(-8, 0, 0));
  } else {
    grid.touch(Vector3i(0, 0, 0));
  }
  // Distributed case
  CHECK(grid.num_active_blocks() == 1);
  grid.fetch_neighbours();
  CHECK(grid.num_active_blocks() == 2);
}

TC_TEST("basic distributed 4") {
  if (!with_mpi())
    return;
  TestGrid grid;
  if (grid.world_size != 4) {
    return;
  }
  grid.touch(Vector3i(-8, 0, 0));
  grid.touch(Vector3i(0, -8, 0));
  grid.touch(Vector3i(-8, -8, 0));
  grid.touch(Vector3i(0, 0, 0));
  // Distributed case
  CHECK(grid.num_active_blocks() == 1);
  grid.fetch_neighbours();
  CHECK(grid.num_active_blocks() == 4);
}

auto test_mpi = [](const std::vector<std::string> &param) {
  if (!with_mpi()) {
    TC_ERROR("Pls execute this task with mpirun");
  }
  MPI_Init(nullptr, nullptr);
  int world_size = 4, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  TC_P(world_rank);
  MPI_Finalize();
};

TC_REGISTER_TASK(test_mpi);

TC_NAMESPACE_END
