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
