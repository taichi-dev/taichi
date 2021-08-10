#pragma once

#include "taichi/common/core.h"
#include "taichi/inc/constants.h"
#include "Eigen/Sparse"

namespace taichi {
namespace lang {

class SparseMatrix {
 public:
  SparseMatrix() {
  }

  void *get_data_pointer() {
    return data.data();
  }

  void *get_num_triplet_pointer() {
    return data.data();
  }

  void build() {
    TI_ASSERT(built == false);
    built = true;
    using T = Eigen::Triplet<float32>;
    std::vector<T> triplets;
    for (int i = 0; i < num_triplets; i++) {
      triplets.push_back({data[i * 3], data[i * 3 + 1],
                          taichi_union_cast<float32>(data[i * 3 + 2])});
    }
    matrix.setFromTriplets(triplets.begin(), triplets.end());
  }

 private:
  std::uint64_t num_triplets{0};
  std::vector<uint32> data;
  bool built{false};
  Eigen::SparseMatrix<float32> matrix;
};

}  // namespace lang
}  // namespace taichi