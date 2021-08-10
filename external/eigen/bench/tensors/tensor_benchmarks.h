#ifndef THIRD_PARTY_EIGEN3_TENSOR_BENCHMARKS_H_
#define THIRD_PARTY_EIGEN3_TENSOR_BENCHMARKS_H_

typedef int TensorIndex;
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include "unsupported/Eigen/CXX11/Tensor"
#include "benchmark.h"

#define BENCHMARK_RANGE(bench, lo, hi) \
  BENCHMARK(bench)->Range(lo, hi)

using Eigen::Tensor;
using Eigen::TensorMap;

// TODO(bsteiner): also templatize on the input type since we have users
// for int8 as well as floats.
template <typename Device, typename T> class BenchmarkSuite {
 public:
  BenchmarkSuite(const Device& device, size_t m, size_t k, size_t n)
      : m_(m), k_(k), n_(n), device_(device) {
    initialize();
  }

  BenchmarkSuite(const Device& device, size_t m)
      : m_(m), k_(m), n_(m), device_(device) {
    initialize();
  }

  ~BenchmarkSuite() {
    device_.deallocate(a_);
    device_.deallocate(b_);
    device_.deallocate(c_);
  }

  void memcpy(int num_iters) {
    eigen_assert(m_ == k_ && k_ == n_);
    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      device_.memcpy(c_, a_, m_ * m_ * sizeof(T));
    }
    // Record the number of values copied per second
    finalizeBenchmark(static_cast<int64_t>(m_) * m_ * num_iters);
  }

  void typeCasting(int num_iters) {
    eigen_assert(m_ == n_);
    Eigen::array<TensorIndex, 2> sizes;
    if (sizeof(T) >= sizeof(int)) {
      sizes[0] = m_;
      sizes[1] = k_;
    } else {
      sizes[0] = m_ * sizeof(T) / sizeof(int);
      sizes[1] = k_ * sizeof(T) / sizeof(int);
    }
    const TensorMap<Tensor<int, 2, 0, TensorIndex>, Eigen::Aligned> A((int*)a_, sizes);
    TensorMap<Tensor<T, 2, 0, TensorIndex>, Eigen::Aligned> B(b_, sizes);

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      B.device(device_) = A.template cast<T>();
    }
    // Record the number of values copied per second
    finalizeBenchmark(static_cast<int64_t>(m_) * k_ * num_iters);
  }

  void random(int num_iters) {
    eigen_assert(m_ == k_ && k_ == n_);
    Eigen::array<TensorIndex, 2> sizes;
    sizes[0] = m_;
    sizes[1] = m_;
    TensorMap<Tensor<T, 2>, Eigen::Aligned> C(c_, sizes);

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = C.random();
    }
    // Record the number of random numbers generated per second
    finalizeBenchmark(static_cast<int64_t>(m_) * m_ * num_iters);
  }

  void slicing(int num_iters) {
    eigen_assert(m_ == k_ && k_ == n_);
    Eigen::array<TensorIndex, 2> sizes;
    sizes[0] = m_;
    sizes[1] = m_;
    const TensorMap<Tensor<T, 2>, Eigen::Aligned> A(a_, sizes);
    const TensorMap<Tensor<T, 2>, Eigen::Aligned> B(b_, sizes);
    TensorMap<Tensor<T, 2>, Eigen::Aligned> C(c_, sizes);

    const Eigen::DSizes<TensorIndex, 2> quarter_sizes(m_/2, m_/2);
    const Eigen::DSizes<TensorIndex, 2> first_quadrant(0, 0);
    const Eigen::DSizes<TensorIndex, 2> second_quadrant(0, m_/2);
    const Eigen::DSizes<TensorIndex, 2> third_quadrant(m_/2, 0);
    const Eigen::DSizes<TensorIndex, 2> fourth_quadrant(m_/2, m_/2);

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.slice(first_quadrant, quarter_sizes).device(device_) =
          A.slice(first_quadrant, quarter_sizes);
      C.slice(second_quadrant, quarter_sizes).device(device_) =
          B.slice(second_quadrant, quarter_sizes);
      C.slice(third_quadrant, quarter_sizes).device(device_) =
          A.slice(third_quadrant, quarter_sizes);
      C.slice(fourth_quadrant, quarter_sizes).device(device_) =
          B.slice(fourth_quadrant, quarter_sizes);
    }
    // Record the number of values copied from the rhs slice to the lhs slice
    // each second
    finalizeBenchmark(static_cast<int64_t>(m_) * m_ * num_iters);
  }

  void rowChip(int num_iters) {
    Eigen::array<TensorIndex, 2> input_size;
    input_size[0] = k_;
    input_size[1] = n_;
    const TensorMap<Tensor<T, 2, 0, TensorIndex>, Eigen::Aligned> B(b_, input_size);
    Eigen::array<TensorIndex, 1> output_size;
    output_size[0] = n_;
    TensorMap<Tensor<T, 1, 0, TensorIndex>, Eigen::Aligned> C(c_, output_size);

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = B.chip(iter % k_, 0);
    }
    // Record the number of values copied from the rhs chip to the lhs.
    finalizeBenchmark(static_cast<int64_t>(n_) * num_iters);
  }

  void colChip(int num_iters) {
    Eigen::array<TensorIndex, 2> input_size;
    input_size[0] = k_;
    input_size[1] = n_;
    const TensorMap<Tensor<T, 2, 0, TensorIndex>, Eigen::Aligned> B(b_, input_size);
    Eigen::array<TensorIndex, 1> output_size;
    output_size[0] = n_;
    TensorMap<Tensor<T, 1, 0, TensorIndex>, Eigen::Aligned> C(c_, output_size);

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = B.chip(iter % n_, 1);
    }
    // Record the number of values copied from the rhs chip to the lhs.
    finalizeBenchmark(static_cast<int64_t>(n_) * num_iters);
  }

  void shuffling(int num_iters) {
    eigen_assert(m_ == n_);
    Eigen::array<TensorIndex, 2> size_a;
    size_a[0] = m_;
    size_a[1] = k_;
    const TensorMap<Tensor<T, 2>, Eigen::Aligned> A(a_, size_a);
    Eigen::array<TensorIndex, 2> size_b;
    size_b[0] = k_;
    size_b[1] = m_;
    TensorMap<Tensor<T, 2>, Eigen::Aligned> B(b_, size_b);

    Eigen::array<int, 2> shuffle;
    shuffle[0] = 1;
    shuffle[1] = 0;

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      B.device(device_) = A.shuffle(shuffle);
    }
    // Record the number of values shuffled from A and copied to B each second
    finalizeBenchmark(static_cast<int64_t>(m_) * k_ * num_iters);
  }

 void padding(int num_iters) {
    eigen_assert(m_ == k_);
    Eigen::array<TensorIndex, 2> size_a;
    size_a[0] = m_;
    size_a[1] = k_-3;
    const TensorMap<Tensor<T, 2>, Eigen::Aligned> A(a_, size_a);
    Eigen::array<TensorIndex, 2> size_b;
    size_b[0] = k_;
    size_b[1] = m_;
    TensorMap<Tensor<T, 2>, Eigen::Aligned> B(b_, size_b);

#if defined(EIGEN_HAS_INDEX_LIST)
    Eigen::IndexPairList<Eigen::type2indexpair<0, 0>,
                         Eigen::type2indexpair<2, 1> > paddings;
#else
    Eigen::array<Eigen::IndexPair<TensorIndex>, 2> paddings;
    paddings[0] = Eigen::IndexPair<TensorIndex>(0, 0);
    paddings[1] = Eigen::IndexPair<TensorIndex>(2, 1);
#endif

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      B.device(device_) = A.pad(paddings);
    }
    // Record the number of values copied from the padded tensor A each second
    finalizeBenchmark(static_cast<int64_t>(m_) * k_ * num_iters);
  }

 void striding(int num_iters) {
    eigen_assert(m_ == k_);
    Eigen::array<TensorIndex, 2> size_a;
    size_a[0] = m_;
    size_a[1] = k_;
    const TensorMap<Tensor<T, 2>, Eigen::Aligned> A(a_, size_a);
    Eigen::array<TensorIndex, 2> size_b;
    size_b[0] = m_;
    size_b[1] = k_/2;
    TensorMap<Tensor<T, 2>, Eigen::Aligned> B(b_, size_b);

#ifndef EIGEN_HAS_INDEX_LIST
    Eigen::array<TensorIndex, 2> strides;
    strides[0] = 1;
    strides[1] = 2;
#else
    // Take advantage of cxx11 to give the compiler information it can use to
    // optimize the code.
    Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2> > strides;
#endif

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      B.device(device_) = A.stride(strides);
    }
    // Record the number of values copied from the padded tensor A each second
    finalizeBenchmark(static_cast<int64_t>(m_) * k_ * num_iters);
  }

  void broadcasting(int num_iters) {
    Eigen::array<TensorIndex, 2> size_a;
    size_a[0] = m_;
    size_a[1] = 1;
    const TensorMap<Tensor<T, 2>, Eigen::Aligned> A(a_, size_a);
    Eigen::array<TensorIndex, 2> size_c;
    size_c[0] = m_;
    size_c[1] = n_;
    TensorMap<Tensor<T, 2>, Eigen::Aligned> C(c_, size_c);

#ifndef EIGEN_HAS_INDEX_LIST
    Eigen::array<int, 2> broadcast;
    broadcast[0] = 1;
    broadcast[1] = n_;
#else
    // Take advantage of cxx11 to give the compiler information it can use to
    // optimize the code.
    Eigen::IndexList<Eigen::type2index<1>, int> broadcast;
    broadcast.set(1, n_);
#endif

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = A.broadcast(broadcast);
    }
    // Record the number of values broadcasted from A and copied to C each second
    finalizeBenchmark(static_cast<int64_t>(m_) * n_ * num_iters);
  }

  void coeffWiseOp(int num_iters) {
    eigen_assert(m_ == k_ && k_ == n_);
    Eigen::array<TensorIndex, 2> sizes;
    sizes[0] = m_;
    sizes[1] = m_;
    const TensorMap<Tensor<T, 2>, Eigen::Aligned> A(a_, sizes);
    const TensorMap<Tensor<T, 2>, Eigen::Aligned> B(b_, sizes);
    TensorMap<Tensor<T, 2>, Eigen::Aligned> C(c_, sizes);

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = A * A.constant(static_cast<T>(3.14)) + B * B.constant(static_cast<T>(2.7));
    }
    // Record the number of FLOP executed per second (2 multiplications and
    // 1 addition per value)
    finalizeBenchmark(static_cast<int64_t>(3) * m_ * m_ * num_iters);
  }

  void algebraicFunc(int num_iters) {
    eigen_assert(m_ == k_ && k_ == n_);
    Eigen::array<TensorIndex, 2> sizes;
    sizes[0] = m_;
    sizes[1] = m_;
    const TensorMap<Tensor<T, 2>, Eigen::Aligned> A(a_, sizes);
    const TensorMap<Tensor<T, 2>, Eigen::Aligned> B(b_, sizes);
    TensorMap<Tensor<T, 2>, Eigen::Aligned> C(c_, sizes);

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = A.rsqrt() + B.sqrt() * B.square();
    }
    // Record the number of FLOP executed per second (assuming one operation
    // per value)
    finalizeBenchmark(static_cast<int64_t>(m_) * m_ * num_iters);
  }

  void transcendentalFunc(int num_iters) {
    eigen_assert(m_ == k_ && k_ == n_);
    Eigen::array<TensorIndex, 2> sizes;
    sizes[0] = m_;
    sizes[1] = m_;
    const TensorMap<Tensor<T, 2>, Eigen::Aligned> A(a_, sizes);
    const TensorMap<Tensor<T, 2>, Eigen::Aligned> B(b_, sizes);
    TensorMap<Tensor<T, 2>, Eigen::Aligned> C(c_, sizes);

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = A.exp() + B.log();
    }
    // Record the number of FLOP executed per second (assuming one operation
    // per value)
    finalizeBenchmark(static_cast<int64_t>(m_) * m_ * num_iters);
  }

 // Row reduction
  void rowReduction(int num_iters) {
    Eigen::array<TensorIndex, 2> input_size;
    input_size[0] = k_;
    input_size[1] = n_;
    const TensorMap<Tensor<T, 2, 0, TensorIndex>, Eigen::Aligned> B(b_, input_size);
    Eigen::array<TensorIndex, 1> output_size;
    output_size[0] = n_;
    TensorMap<Tensor<T, 1, 0, TensorIndex>, Eigen::Aligned> C(c_, output_size);

#ifndef EIGEN_HAS_INDEX_LIST
    Eigen::array<TensorIndex, 1> sum_along_dim;
    sum_along_dim[0] = 0;
#else
    // Take advantage of cxx11 to give the compiler information it can use to
    // optimize the code.
    Eigen::IndexList<Eigen::type2index<0>> sum_along_dim;
#endif

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = B.sum(sum_along_dim);
    }
    // Record the number of FLOP executed per second (assuming one operation
    // per value)
    finalizeBenchmark(static_cast<int64_t>(k_) * n_ * num_iters);
  }

  // Column reduction
  void colReduction(int num_iters) {
    Eigen::array<TensorIndex, 2> input_size;
    input_size[0] = k_;
    input_size[1] = n_;
    const TensorMap<Tensor<T, 2, 0, TensorIndex>, Eigen::Aligned> B(
        b_, input_size);
    Eigen::array<TensorIndex, 1> output_size;
    output_size[0] = k_;
    TensorMap<Tensor<T, 1, 0, TensorIndex>, Eigen::Aligned> C(
        c_, output_size);

#ifndef EIGEN_HAS_INDEX_LIST
    Eigen::array<TensorIndex, 1> sum_along_dim;
    sum_along_dim[0] = 1;
#else
    // Take advantage of cxx11 to give the compiler information it can use to
    // optimize the code.
    Eigen::IndexList<Eigen::type2index<1>> sum_along_dim;
#endif

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = B.sum(sum_along_dim);
    }
    // Record the number of FLOP executed per second (assuming one operation
    // per value)
    finalizeBenchmark(static_cast<int64_t>(k_) * n_ * num_iters);
  }

  // Full reduction
  void fullReduction(int num_iters) {
    Eigen::array<TensorIndex, 2> input_size;
    input_size[0] = k_;
    input_size[1] = n_;
    const TensorMap<Tensor<T, 2, 0, TensorIndex>, Eigen::Aligned> B(
        b_, input_size);
    Eigen::array<TensorIndex, 0> output_size;
    TensorMap<Tensor<T, 0, 0, TensorIndex>, Eigen::Aligned> C(
        c_, output_size);

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = B.sum();
    }
    // Record the number of FLOP executed per second (assuming one operation
    // per value)
    finalizeBenchmark(static_cast<int64_t>(k_) * n_ * num_iters);
  }

  // do a contraction which is equivalent to a matrix multiplication
  void contraction(int num_iters) {
    Eigen::array<TensorIndex, 2> sizeA;
    sizeA[0] = m_;
    sizeA[1] = k_;
    Eigen::array<TensorIndex, 2> sizeB;
    sizeB[0] = k_;
    sizeB[1] = n_;
    Eigen::array<TensorIndex, 2> sizeC;
    sizeC[0] = m_;
    sizeC[1] = n_;

    const TensorMap<Tensor<T, 2>, Eigen::Aligned> A(a_, sizeA);
    const TensorMap<Tensor<T, 2>, Eigen::Aligned> B(b_, sizeB);
    TensorMap<Tensor<T, 2>, Eigen::Aligned> C(c_, sizeC);

    typedef typename Tensor<T, 2>::DimensionPair DimPair;
    Eigen::array<DimPair, 1> dims;
    dims[0] = DimPair(1, 0);

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = A.contract(B, dims);
    }
    // Record the number of FLOP executed per second (size_ multiplications and
    // additions for each value in the resulting tensor)
    finalizeBenchmark(static_cast<int64_t>(2) * m_ * n_ * k_ * num_iters);
  }

  void convolution(int num_iters, int kernel_x, int kernel_y) {
    Eigen::array<TensorIndex, 2> input_sizes;
    input_sizes[0] = m_;
    input_sizes[1] = n_;
    TensorMap<Tensor<T, 2>, Eigen::Aligned> A(a_, input_sizes);
    Eigen::array<TensorIndex, 2> kernel_sizes;
    kernel_sizes[0] = kernel_x;
    kernel_sizes[1] = kernel_y;
    TensorMap<Tensor<T, 2>, Eigen::Aligned> B(b_, kernel_sizes);
    Eigen::array<TensorIndex, 2> result_sizes;
    result_sizes[0] = m_ - kernel_x + 1;
    result_sizes[1] = n_ - kernel_y + 1;
    TensorMap<Tensor<T, 2>, Eigen::Aligned> C(c_, result_sizes);
    Eigen::array<TensorIndex, 2> dims;
    dims[0] = 0;
    dims[1] = 1;

    StartBenchmarkTiming();
    for (int iter = 0; iter < num_iters; ++iter) {
      C.device(device_) = A.convolve(B, dims);
    }
    // Record the number of FLOP executed per second (kernel_size
    // multiplications and additions for each value in the resulting tensor)
    finalizeBenchmark(static_cast<int64_t>(2) *
        (m_ - kernel_x + 1) * (n_ - kernel_y + 1) * kernel_x * kernel_y * num_iters);
  }

 private:
  void initialize() {
    a_ = (T *) device_.allocate(m_ * k_ * sizeof(T));
    b_ = (T *) device_.allocate(k_ * n_ * sizeof(T));
    c_ = (T *) device_.allocate(m_ * n_ * sizeof(T));

    // Initialize the content of the memory pools to prevent asan from
    // complaining.
    device_.memset(a_, 12, m_ * k_ * sizeof(T));
    device_.memset(b_, 23, k_ * n_ * sizeof(T));
    device_.memset(c_, 31, m_ * n_ * sizeof(T));

    //BenchmarkUseRealTime();
  }

  inline void finalizeBenchmark(int64_t num_items) {
#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
    if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
      device_.synchronize();
    }
#endif
    StopBenchmarkTiming();
    SetBenchmarkFlopsProcessed(num_items);
  }


  TensorIndex m_;
  TensorIndex k_;
  TensorIndex n_;
  T* a_;
  T* b_;
  T* c_;
  Device device_;
};
#endif  // THIRD_PARTY_EIGEN3_TENSOR_BENCHMARKS_H_
