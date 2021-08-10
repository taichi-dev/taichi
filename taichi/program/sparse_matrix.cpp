#include "taichi/program/sparse_matrix.h"

namespace taichi {
namespace lang {

SparseMatrix::SparseMatrix(int n, int m, int max_num_triplets) : n(n), m(m), max_num_triplets(max_num_triplets) {
  data.reserve(max_num_triplets * 3);
  data_base_ptr = get_data_base_ptr();
}

void *SparseMatrix::get_data_base_ptr() {
  return data.data();
}

void SparseMatrix::build() {
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

}
}