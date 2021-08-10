#include "taichi/program/sparse_matrix.h"

#include "Eigen/Dense"

namespace taichi {
namespace lang {

SparseMatrix::SparseMatrix(int n, int m, int max_num_triplets) : n(n), m(m), max_num_triplets(max_num_triplets), matrix(n, m) {
  data.reserve(max_num_triplets * 3);
  data_base_ptr = get_data_base_ptr();
}

void *SparseMatrix::get_data_base_ptr() {
  return data.data();
}

void SparseMatrix::print_triplets() {
  printf("n=%d, m=%d, num_triplets=%lld (max=%lld)\n", n, m, num_triplets, max_num_triplets);
  for (int64 i = 0; i < num_triplets; i++) {
    printf("(%d, %d) val=%f\n", data[i * 3], data[i * 3 + 1], taichi_union_cast<float32>(data[i * 3 + 2]));
  }
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

void SparseMatrix::print() {
  TI_ASSERT(built = true);
  Eigen::IOFormat clean_fmt(4, 0, ", ", "\n", "[", "]");
  // Note that the code below first converts the sparse matrix into a dense one.
  // https://stackoverflow.com/questions/38553335/how-can-i-print-in-console-a-formatted-sparse-matrix-with-eigen
  std::cout << Eigen::MatrixXf(matrix).format(clean_fmt) << std::endl;
}

}
}