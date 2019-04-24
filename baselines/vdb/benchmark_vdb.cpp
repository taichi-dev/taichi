#include "../../src/tlang.h"
#include <taichi/testing.h>
#include <taichi/system/timer.h>
#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/tools/ChangeBackground.h>

TLANG_NAMESPACE_BEGIN

TC_TEST("hashed_3d") {
  return;
  CoreState::set_trigger_gdb_when_crash(true);

  Program prog;

  Global(x, i32);
  Global(sum, i32);

  int n = 256;

  layout([&] {
    auto i = Index(0), j = Index(1), k = Index(2);
    root.hashed({i, j, k}, n).place(x);
    root.place(sum);
  });

  x.val<int>(2, 5, 9) = 10;
  x.val<int>(12, 5, 9) = 30;
  x.val<int>(2, 115, 9) = 11;

  kernel([&] {
    Declare(i);
    Declare(j);
    Declare(k);
    For((i, j, k), x, [&]() { sum[Expr(0)] += x[i, j, k]; });
  })();

  TC_ASSERT(sum.val<int>() == 51);
};

TC_TEST("hashed_3d_negative") {
  return;
  CoreState::set_trigger_gdb_when_crash(true);

  Program prog;

  Global(x, i32);
  Global(sum, i32);

  int n = 256;

  layout([&] {
    auto i = Index(0), j = Index(1), k = Index(2);
    root.hashed({i, j, k}, n).place(x);
    root.place(sum);
  });

  x.val<int>(-2, 5, 9) = 10;
  x.val<int>(12, 5, -9) = 30;
  x.val<int>(2, -115, 9) = 11;

  kernel([&] {
    Declare(i);
    Declare(j);
    Declare(k);
    For((i, j, k), x, [&]() { sum[Expr(0)] += x[i, j, k]; });
  })();

  TC_ASSERT(sum.val<int>() == 51);
  TC_CHECK(x.val<int>(-2, 5, 9) == 10);
  TC_CHECK(x.val<int>(12, 5, -9) == 30);
  TC_CHECK(x.val<int>(2, -115, 9) == 11);
};

auto benchmark_vdb = [](std::vector<std::string> param) {
  TC_ASSERT(param.size() == 1);
  auto fn = "vdb_dataset/" + param[0];

  CoreState::set_trigger_gdb_when_crash(true);
  Program prog;

  AmbientGlobal(x, f32, 0.0f);
  Global(y, f32);
  Global(sum, i32);

  int tree_config[] = {5, 4, 3};

  for (int i = 0; i < 3; i++) {
    tree_config[i] = 1 << tree_config[i];
  }

  bool AOS = true;

  layout([&] {
    auto ijk = Indices(0, 1, 2);

    if (AOS) {
      root.hashed(ijk, 1024)
          .dense(ijk, tree_config[0])
          .pointer()
          .dense(ijk, tree_config[1])
          .pointer()
          .dense(ijk, tree_config[2])
          .place(x, y);
    } else {
      auto &fork = root.hashed(ijk, 1024)
                       .dense(ijk, tree_config[0])
                       .pointer()
                       .dense(ijk, tree_config[1])
                       .pointer();

      fork.dense(ijk, tree_config[2]).place(x);

      fork.dense(ijk, tree_config[2]).place(y);
    }

    root.place(sum);
  });

  int num_leaves = 0;

  using GridType = openvdb::FloatGrid;
  using TreeType = GridType::TreeType;
  using RootType = TreeType::RootNodeType;  // level 3 RootNode
  TC_ASSERT(RootType::LEVEL == 3);
  using Int1Type = RootType::ChildNodeType;  // level 2 InternalNode
  using Int2Type = Int1Type::ChildNodeType;  // level 1 InternalNode
  using LeafType = TreeType::LeafNodeType;   // level 0 LeafNode

  // Initialize the OpenVDB library.  This must be called at least
  // once per program and may safely be called multiple times.
  openvdb::initialize();
  // Create an empty floating-point grid with background value 0.

  std::ifstream ifile(fn, std::ios_base::binary);
  auto grids = openvdb::io::Stream(ifile).getGrids();

  auto grid = static_cast<GridType *>((*grids)[0].get());
  openvdb::tools::changeBackground(grid->tree(), 0.0f);

  auto dsl_value = [&](Expr var, openvdb::Coord coord) -> float32 & {
    int offset = 0;
    int i = coord.x() + offset, j = coord.y() + offset, k = coord.z() + offset;

    return var.val<float32>(i, j, k);
  };

  // densely fill voxel
  for (auto iter = grid->tree().beginLeaf(); iter; ++iter) {
    auto &leaf = *iter;
    for (int t = 0; t < 512; t++) {
      real value;
      if (!leaf.probeValue(t, value)) {
        leaf.setValueOn(t, 1.0f);
      }
      auto coord = leaf.offsetToGlobalCoord(t);
      dsl_value(x, coord) = leaf.getValue(t);
    }
    num_leaves += 1;
  }

  TC_P(grid->tree().getValue(openvdb::Coord(-288, -9, -23)));

  int counter[4] = {0};
  for (TreeType::NodeIter iter = grid->tree().beginNode(); iter; ++iter) {
    switch (iter.getDepth()) {
      case 0: {
        RootType *node = nullptr;
        iter.getNode(node);
        if (node)
          counter[0]++;
        break;
      }
      case 1: {
        Int1Type *node = nullptr;
        iter.getNode(node);
        if (node)
          counter[1]++;
        break;
      }
      case 2: {
        Int2Type *node = nullptr;
        iter.getNode(node);
        if (node)
          counter[2]++;
        break;
      }
      case 3: {
        LeafType *node = nullptr;
        iter.getNode(node);
        if (node)
          counter[3]++;
        break;
      }
    }
  }
  TC_P(grid->activeVoxelCount());
  for (int i = 0; i < 4; i++) {
    TC_INFO("Depth {}: nodes {}", i, counter[i]);
  }

  openvdb::tools::Filter<GridType> filter(*grid);

  tbb::task_arena limited(1);

  limited.execute([&] {
    for (int i = 0; i < 1; i++)
      TC_TIME(filter.mean(1));
  });

  auto &count = kernel([&] {
    Declare(i);
    Declare(j);
    Declare(k);
    For((i, j, k), x, [&]() { sum[Expr(0)] += 1; });
  });

  count();

  TC_P(num_leaves);
  // TC_ASSERT(num_leaves * pow<3>(8) == sum.val<int>());

  auto &mean_x = kernel([&] {
    Declare(i);
    Declare(j);
    Declare(k);
    For((i, j, k), x, [&]() {
      y[i, j, k] = (1.0_f / 3) * (x[i - 1, j, k] + x[i, j, k] + x[i + 1, j, k]);
    });
  });

  auto &mean_y = kernel([&] {
    Declare(i);
    Declare(j);
    Declare(k);
    For((i, j, k), x, [&]() {
      y[i, j, k] = (1.0_f / 3) * (x[i, j - 1, k] + x[i, j, k] + x[i, j + 1, k]);
    });
  });

  auto &mean_z = kernel([&] {
    Declare(i);
    Declare(j);
    Declare(k);
    For((i, j, k), x, [&]() {
      y[i, j, k] = (1.0_f / 3) * (x[i, j, k - 1] + x[i, j, k] + x[i, j, k + 1]);
    });
  });

  auto &copy_y_to_x = kernel([&] {
    Declare(i);
    Declare(j);
    Declare(k);
    For((i, j, k), x, [&]() { x[i, j, k] = y[i, j, k]; });
  });

  auto mean = [&]() {
    mean_x();
    copy_y_to_x();
    mean_y();
    copy_y_to_x();
    mean_z();
    copy_y_to_x();
  };

  for (int i = 0; i < 1; i++)
    TC_TIME(mean());

  // densely fill voxel
  for (auto iter = grid->tree().beginLeaf(); iter; ++iter) {
    auto &leaf = *iter;
    for (int t = 0; t < 512; t++) {
      // TC_P(t);
      auto coord = leaf.offsetToGlobalCoord(t);
      // fmt::print("{} {} {}\n", coord.x(), coord.y(), coord.z());
      auto dsl = dsl_value(x, coord), vdb = leaf.getValue(t);
      TC_ASSERT_EQUAL(dsl, vdb, 4e-5_f);
    }
    num_leaves += 1;
  }

  std::ofstream ofs(fmt::format("results/{}.txt", param[0]),
                    std::ofstream::out);

  TC_ASSERT(ofs.is_open());

  int N = 3;
  auto t = Time::get_time();
  for (int i = 0; i < N; i++) {
    TC_TIME(mean());
  }
  ofs << Time::get_time() - t << std::endl;

  t = Time::get_time();
  limited.execute([&] {
    for (int i = 0; i < N; i++)
      TC_TIME(filter.mean(1));
  });
  ofs << Time::get_time() - t << std::endl;
};

TC_REGISTER_TASK(benchmark_vdb);

TLANG_NAMESPACE_END
