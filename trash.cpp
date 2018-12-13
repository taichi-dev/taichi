// PartI:
/*
constexpr int start = Scratch::linear_offset<-1, -1, -1>();
constexpr int end = Scratch::linear_offset<Block::size[0],
Block::size[1],
                                       Block::size[2]>();
// TODO: there is a +1 missing after Block::size!!

for (int p = start; p < end; p += 8) {
  __m256 sum_z = _mm256_add_ps(
      _mm256_loadu_ps(&scratchU.linearized_data
                           [p + Scratch::relative_offset<0, 0,
-1>()]),
      _mm256_loadu_ps(&scratchU.linearized_data
                           [p + Scratch::relative_offset<0, 0,
1>()]));
  __m256 sum_y = _mm256_add_ps(
      _mm256_loadu_ps(&scratchU.linearized_data
                           [p + Scratch::relative_offset<0, -1,
0>()]),
      _mm256_loadu_ps(&scratchU.linearized_data
                           [p + Scratch::relative_offset<0, 1,
0>()]));
  __m256 sum_x = _mm256_add_ps(
      _mm256_loadu_ps(&scratchU.linearized_data
                           [p + Scratch::relative_offset<-1, 0,
0>()]),
      _mm256_loadu_ps(&scratchU.linearized_data
                           [p + Scratch::relative_offset<1, 0,
0>()]));

  auto sum = _mm256_add_ps(
      _mm256_add_ps(sum_z, sum_y),
      _mm256_add_ps(sum_x,
                    _mm256_loadu_ps(&scratchB.linearized_data[p])));
  auto original = _mm256_loadu_ps(&scratchU.linearized_data[p]);

  // o = original + (tmp * (1.0_f / 6) - original) * (2.0_f / 3_f);
  sum = _mm256_add_ps(
      original,
      _mm256_mul_ps(
          _mm256_sub_ps(_mm256_mul_ps(sum, _mm256_set1_ps(1.0f / 6)),
                        original),
          _mm256_set1_ps(2.0f / 3)));
  _mm256_storeu_ps(&scratchV.linearized_data[p], sum);
}
*/

// PartII:
/*)
for (int i = 0; i < Block::size[0]; i++) {
  for (int j = 0; j < Block::size[1]; j++) {
    __m256 sum_z =
        _mm256_add_ps(_mm256_loadu_ps(&scratchV.data[i][j][-1]),
                      _mm256_loadu_ps(&scratchV.data[i][j][1]));
    __m256 sum_y =
        _mm256_add_ps(_mm256_loadu_ps(&scratchV.data[i][j - 1][0]),
                      _mm256_loadu_ps(&scratchV.data[i][j + 1][0]));
    __m256 sum_x =
        _mm256_add_ps(_mm256_loadu_ps(&scratchV.data[i - 1][j][0]),
                      _mm256_loadu_ps(&scratchV.data[i + 1][j][0]));

    auto sum = _mm256_add_ps(
        _mm256_add_ps(sum_z, sum_y),
        _mm256_add_ps(sum_x,
                      _mm256_loadu_ps(&scratchB.data[i][j][0])));
    auto original = _mm256_loadu_ps(&scratchV.data[i][j][0]);

    // o = original + (tmp * (1.0_f / 6) - original) * (2.0_f / 3_f);
    sum = _mm256_add_ps(
        original,
        _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(sum, _mm256_set1_ps(
                                                           1.0f / 6)),
                                    original),
                      _mm256_set1_ps(2.0f / 3)));
    _mm256_storeu_ps(&b.node_local(Vector3i(i, j, 0))[U], sum);
    // (B - Lu) / Diag
    // Damping is important. It brings down #iterations to 1e-7 from
    // 91 to 10...
  }
}

  void smooth_single(int level, int U, int B) {
    TC_PROFILER("smoothing")
    // TODO: this supports zero-Dirichlet BC only!
    grids[level]->advance(
        [&](Grid::Block &b, Grid::Ancestors &an) {
          if (!b.meta.get_has_effective_cell())
            return;
          GridScratchPadCh scratchB(an, B * sizeof(real));
          GridScratchPadCh scratchU(an, U * sizeof(real));
          // 6 neighbours
          TC_STATIC_ASSERT(sizeof(real) == 4);
          TC_STATIC_ASSERT(Block::size[2] == 8);
          for (int i = 0; i < Block::size[0]; i++) {
            for (int j = 0; j < Block::size[1]; j++) {
              __m256 sum_z =
                  _mm256_add_ps(_mm256_loadu_ps(&scratchU.data[i][j][-1]),
                                _mm256_loadu_ps(&scratchU.data[i][j][1]));
              __m256 sum_y =
                  _mm256_add_ps(_mm256_loadu_ps(&scratchU.data[i][j - 1][0]),
                                _mm256_loadu_ps(&scratchU.data[i][j + 1][0]));
              __m256 sum_x =
                  _mm256_add_ps(_mm256_loadu_ps(&scratchU.data[i - 1][j][0]),
                                _mm256_loadu_ps(&scratchU.data[i + 1][j][0]));

              auto sum = _mm256_add_ps(
                  _mm256_add_ps(sum_z, sum_y),
                  _mm256_add_ps(sum_x,
                                _mm256_loadu_ps(&scratchB.data[i][j][0])));
              auto original = _mm256_loadu_ps(&scratchU.data[i][j][0]);

              // o = original + (tmp * (1.0_f / 6) - original) * (2.0_f / 3_f);
              sum = _mm256_add_ps(
                  original,
                  _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(sum, _mm256_set1_ps(
                                                                     1.0f / 6)),
                                              original),
                                _mm256_set1_ps(2.0f / 3)));
              _mm256_storeu_ps(&b.node_local(Vector3i(i, j, 0))[U], sum);
              // (B - Lu) / Diag
              // Damping is important. It brings down #iterations to 1e-7 from
              // 91 to 10...
            }
          }
        },
        false, level == 0);  // carry nodes only if on finest level
  }
*/

// Residual
/*
// 6 neighbours
for (int i = 0; i < Block::size[0]; i++) {
  for (int j = 0; j < Block::size[1]; j++) {
    for (int k = 0; k < Block::size[2]; k++) {
      auto rhs = b.node_local(Vector3i(i, j, k))[B];
      auto c = b.node_local(Vector3i(i, j, k))[U];
      auto fetch = [&](int ii, int jj, int kk) {
        rhs += (scratch.data[i + ii][j + jj][k + kk] - c);
      };
      fetch(0, 0, 1);
      fetch(0, 0, -1);
      fetch(0, 1, 0);
      fetch(0, -1, 0);
      fetch(1, 0, 0);
      fetch(-1, 0, 0);
      b.node_local(Vector3i(i, j, k))[R] = rhs;
    }
  }
}
*/

// Multiply
/*
// 6 neighbours
for (int i = 0; i < Block::size[0]; i++) {
  for (int j = 0; j < Block::size[1]; j++) {
    for (int k = 0; k < Block::size[2]; k++) {
      int count = 0;
      real tmp = 0;
      auto fetch = [&](int ii, int jj, int kk) {
        auto &n = scratch.data[i + (ii)][j + (jj)][k + (kk)];
        count++;
        tmp += n;
      };
      fetch(0, 0, 1);
      fetch(0, 0, -1);
      fetch(0, 1, 0);
      fetch(0, -1, 0);
      fetch(1, 0, 0);
      fetch(-1, 0, 0);
      auto &o = b.node_local(Vector3i(i, j, k))[channel_out];
      o = 6 * scratch.data[i][j][k] - tmp;
    }
  }
}
*/




// -----------------------------------------------------------------------------


/*
template <int dim, typename T>
real AOS2_matmatmul() {
  struct Mat {
    T d[dim][dim];
  };
  std::vector<Mat> A, B, C;
  A.resize(N);
  B.resize(N);
  C.resize(N);

  auto t = Time::get_time();
  for (int r = 0; r < rounds; r++) {
    for (int t = 0; t < N; t++) {
      Mat X, Y;
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          X.d[i][j] = A[t].d[i][j];
          Y.d[i][j] = B[t].d[i][j];
        }
      }
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          T sum = 0;
          for (int k = 0; k < dim; k++) {
            sum += X.d[i][k] * Y.d[k][j];
          }
          C[t].d[i][j] = sum;
        }
      }
    }
  }
  return Time::get_time() - t;
}
*/



/*
auto test_saxpy = []() {
  // fmt::print("dim={} {} in_cache={} unroll={} prefetch={:2d} ",);
  using namespace Tlang;
  CodeGen cg;
  auto alloc = cg.alloc;
  auto &buffer = alloc.buffer(0);

  int64 enlarge = 4096;
  int64 n = taichi::N * enlarge;
  int64 rounds = taichi::rounds / enlarge;
  cg.unroll = 4;
  cg.prefetch = 0;

  Expr ret;
  alloc.buffer(1).stream(0).place()

  AlignedAllocator A_allocator(n * sizeof(float32)),
      B_allocator(n * sizeof(float32));

  auto func = cg.get(ret);
  for (int i = 0; i < 10; i++)
    func(M_allocator.get<float32>(), V_allocator.get<float32>(),
         MV_allocator.get<float32>(), n);

  for (int K = 0; K < 1; K++) {
    float64 t = Time::get_time();
    for (int i = 0; i < rounds; i++) {
      func(M_allocator.get<float32>(), V_allocator.get<float32>(),
           MV_allocator.get<float32>(), n);
    }
    print_time(Time::get_time() - t, n * rounds);
  }

  for (int i = 0; i < n; i++) {
    auto computed = MV_allocator.get<float32>()[mv(j)->addr.eval(i, n)];
  }
};
*/


// array of N * dim * dim * 8 * float64
/*
template <int dim>
void AOSOA_matmul(float64 *A, float64 *B, float64 *C) {
  constexpr int simd_width = 4;
  for (int r = 0; r < rounds; r++) {
    for (int t = 0; t < N / simd_width; t++) {
      __m256d a[dim * dim], b[dim * dim];
      const int p = dim * dim * simd_width * t;
      for (int i = 0; i < dim * dim; i++) {
        a[i] = _mm256_load_pd(&A[p + simd_width * i]);
        b[i] = _mm256_load_pd(&B[p + simd_width * i]);
      }
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          __m256d c = a[i * dim] * b[j];
          for (int k = 1; k < dim; k++) {
            c = c + a[i * dim + k] * b[k * dim + j];
            // c = _mm256_fmadd_ps(a[i * dim + k], b[k * dim + j], c);
          }
          _mm256_store_pd(&C[p + simd_width * (i * dim + j)], c);
        }
      }
    }
  }
}
*/


/*
// array of N * dim * dim * 8 * float64
template <int dim>
void SOA_matmul(float64 *A, float64 *B, float64 *C) {
  constexpr int simd_width = 4;
  for (int r = 0; r < rounds; r++) {
    for (int t = 0; t < N / simd_width; t++) {
      __m256d a[dim * dim], b[dim * dim];
      for (int i = 0; i < dim * dim; i++) {
        a[i] = _mm256_load_pd(&A[i * N + t * simd_width]);
        b[i] = _mm256_load_pd(&B[i * N + t * simd_width]);
      }
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          __m256d c = a[i * dim] * b[j];
          for (int k = 1; k < dim; k++) {
            c = c + a[i * dim + k] * b[k * dim + j];
            // c = _mm256_fmadd_ps(a[i * dim + k], b[k * dim + j], c);
          }
          _mm256_store_pd(&C[(i * dim + j) * N + t * simd_width], c);
        }
      }
    }
  }
}
*/

