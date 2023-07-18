#include "gtest/gtest.h"

#include "taichi/struct/snode_tree.h"

namespace taichi::lang {

TEST(SNodeTree, GetSNodeToRootMapping) {
  constexpr int kSNodeSize = 16;
  SNode root{/*depth=*/0, /*t=*/SNodeType::root};
  const std::vector<Axis> axes = {Axis{0}};
  std::vector<int> all_snode_ids;
  for (int i = 0; i < 3; ++i) {
    auto &ptr_snode = root.pointer(axes, kSNodeSize);
    auto &dense_snode = ptr_snode.dense(axes, kSNodeSize);
    auto &leaf_snode = dense_snode.insert_children(SNodeType::place);
    all_snode_ids.push_back(ptr_snode.id);
    all_snode_ids.push_back(dense_snode.id);
    all_snode_ids.push_back(leaf_snode.id);
  }

  auto map = get_snodes_to_root_id(root);
  EXPECT_EQ(map.size(), 1 + 3 * 3);
  for (int id : all_snode_ids) {
    EXPECT_EQ(map.at(id), root.id);
  }
}

}  // namespace taichi::lang
