// vim: ft=glsl
// clang-format off
#include "taichi/util/macros.h"
STR(
int fast_pow_i32(int x, int y)
{
  if (y > 512)
    return int(pow(x, y));

  bool neg = y < 0;
  y = abs(y);
  int ret = 1;
  while (y != 0) {
    if ((y & 1) != 0)
      ret *= x;
    x *= x;
    y >>= 1;
  }
  return neg ? 1 / ret : ret;
}

float fast_pow_f32(float x, int y)
{
  if (y > 512)
    return pow(x, y);

  bool neg = y < 0;
  y = abs(y);
  float ret = 1.0;
  while (y != 0) {
    if ((y & 1) != 0)
      ret *= x;
    x *= x;
    y >>= 1;
  }
  return neg ? 1.0 / ret : ret;
}
)
