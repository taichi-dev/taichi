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



