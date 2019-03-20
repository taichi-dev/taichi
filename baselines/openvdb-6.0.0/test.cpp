#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include <openvdb/tools/Filter.h>
#include <iostream>
#include <tbb/tbb.h>
#include "taichi.h"

int main(int argc, char **argv) {
  using GridType = openvdb::FloatGrid;
  using TreeType = GridType::TreeType;
  using RootType = TreeType::RootNodeType;  // level 3 RootNode
  assert(RootType::LEVEL == 3);
  using Int1Type = RootType::ChildNodeType;  // level 2 InternalNode
  using Int2Type = Int1Type::ChildNodeType;  // level 1 InternalNode
  using LeafType = TreeType::LeafNodeType;   // level 0 LeafNode

  // Initialize the OpenVDB library.  This must be called at least
  // once per program and may safely be called multiple times.
  openvdb::initialize();
  // Create an empty floating-point grid with background value 0.

  std::ifstream ifile("dataset/" + std::string(argv[1]), std::ios_base::binary);
  auto grids = openvdb::io::Stream(ifile).getGrids();
  TC_P(grids->size());

  auto grid = static_cast<GridType *>((*grids)[0].get());

  // densely fill voxel
  for (auto iter = grid->tree().beginLeaf(); iter; ++iter) {
    auto &leaf = *iter;
    for (int i = 0; i < 512; i++) {
      leaf.setValueOn(i, 0.0f);
    }
  }

  std::FILE *file = std::fopen("blocks.txt", "w");

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
        auto coord = node->offsetToGlobalCoord(0);
        fprintf(file, "%d %d %d\n", coord.x(), coord.y(), coord.z());
        break;
      }
    }
  }
  fclose(file);
  for (int i = 0; i < 4; i++) {
    TC_INFO("Depth {}: nodes {}", i, counter[i]);
  }
  TC_P(grid->activeVoxelCount());

  openvdb::tools::Filter<GridType> filter(*grid);

  for (int i = 0; i < 10000000; i++)
    TC_TIME(filter.mean(1));

  /*
  // Set the voxel value at (1000, -200000000, 30000000) to 1.
  accessor.setValue(xyz, 1.0);
  // Verify that the voxel value at (1000, -200000000, 30000000) is 1.
  std::cout << "Grid" << xyz << " = " << accessor.getValue(xyz) << std::endl;
  // Reset the coordinates to those of a different voxel.
  xyz.reset(1000, 200000000, -30000000);
  // Verify that the voxel value at (1000, 200000000, -30000000) is
  // the background value, 0.
  std::cout << "Grid" << xyz << " = " << accessor.getValue(xyz) << std::endl;
  // Set the voxel value at (1000, 200000000, -30000000) to 2.
  accessor.setValue(xyz, 2.0);
  // Set the voxels at the two extremes of the available coordinate space.
  // For 32-bit signed coordinates these are (-2147483648, -2147483648,
  // -2147483648) and (2147483647, 2147483647, 2147483647).
  accessor.setValue(openvdb::Coord::min(), 3.0f);
  accessor.setValue(openvdb::Coord::max(), 4.0f);
  std::cout << "Testing sequential access:" << std::endl;
  // Print all active ("on") voxels by means of an iterator.
  for (openvdb::FloatGrid::ValueOnCIter iter = grid->cbeginValueOn(); iter;
       ++iter) {
    std::cout << "Grid" << iter.getCoord() << " = " << *iter << std::endl;
  }
  */
}
