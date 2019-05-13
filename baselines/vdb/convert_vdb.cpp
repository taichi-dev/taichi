#include <taichi/testing.h>
#include <taichi/system/timer.h>
#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/tools/ChangeBackground.h>
#include <taichi/common/task.h>

TC_NAMESPACE_BEGIN

auto convert_vdb = [](std::vector<std::string> param) {
  TC_ASSERT(param.size() == 1);
  auto fn = "vdb_dataset/" + param[0];

  CoreState::set_trigger_gdb_when_crash(true);

  int tree_config[] = {5, 4, 3};

  for (int i = 0; i < 3; i++) {
    tree_config[i] = 1 << tree_config[i];
  }

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

  int min_bounds[3], max_bounds[3], box_size[3];
  for (int i = 0; i < 3; i++) {
    min_bounds[i] = 1000000;
    max_bounds[i] = -1000000;
  }
  real min_value = 1e10;
  real max_value = -1e10;

  // densely fill voxel
  for (auto iter = grid->tree().beginLeaf(); iter; ++iter) {
    auto &leaf = *iter;
    for (int t = 0; t < 512; t++) {
      auto coord = leaf.offsetToGlobalCoord(t);
      // std::cout << coord << std::endl;
      // std::cout << leaf.getValue(t) << std::endl;
      for (int i = 0; i < 3; i++) {
        min_bounds[i] = std::min(min_bounds[i], coord[i]);
        max_bounds[i] = std::max(max_bounds[i], coord[i] + 1);
      }
      min_value = std::min(min_value, leaf.getValue(t));
      max_value = std::max(max_value, leaf.getValue(t));
    }
    num_leaves += 1;
  }
  TC_P(min_bounds);
  TC_P(max_bounds);
  for (int i = 0; i < 3; i++) {
    box_size[i] = max_bounds[i] - min_bounds[i];
    TC_P(box_size[i]);
  }
  TC_P(min_value);
  TC_P(max_value);

  std::vector<float32> density(box_size[0] * box_size[1] * box_size[2], 0.0f);
  // densely fill voxel
  for (auto iter = grid->tree().beginLeaf(); iter; ++iter) {
    auto &leaf = *iter;
    for (int t = 0; t < 512; t++) {
      auto coord = leaf.offsetToGlobalCoord(t);
      real value = leaf.getValue(t);
      density[(coord[0] - min_bounds[0]) * (box_size[2] * box_size[1]) +
              (coord[1] - min_bounds[1]) * (box_size[2]) +
              (coord[2] - min_bounds[2])] = value;
    }
  }

  auto f = fopen("bunny_cloud.bin", "wb");
  fwrite(density.data(), density.size(), sizeof(float32), f);
  fclose(f);
};

/*
[D 05/13/19 15:50:44.839] [convert_vdb.cpp:operator()@68]
min_bounds: [-304, -48, -208]
[D 05/13/19 15:50:44.839] [convert_vdb.cpp:operator()@69]
max_bounds: [280, 528, 232]
[D 05/13/19 15:50:44.839] [convert_vdb.cpp:operator()@72]
box_size[i]: 584
[D 05/13/19 15:50:44.839] [convert_vdb.cpp:operator()@72]
box_size[i]: 576
[D 05/13/19 15:50:44.839] [convert_vdb.cpp:operator()@72]
box_size[i]: 440
[D 05/13/19 15:50:44.839] [convert_vdb.cpp:operator()@74]
min_value: 0
[D 05/13/19 15:50:44.839] [convert_vdb.cpp:operator()@75]
max_value: 2.7923
*/


TC_REGISTER_TASK(convert_vdb);

TC_NAMESPACE_END
