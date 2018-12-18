#include "tlang.h"

TC_NAMESPACE_BEGIN

using namespace Tlang;

/*
auto advection = []() {
  int n = 128, nattr = 4;

  Float attr[4], v[2];  // ; vx, vy

  Program prog;

  for (int i = 0; i < 4; i++) {
    prog.buffer(0).stream(0).group(0).place(attr[i]);
  }
  prog.buffer(0).stream(1).group(0).place(v[0], v[1]);

  // TODO: inv_dx, t

  auto offset_x = floor(v[0]);
  auto offset_y = floor(v[1]);
  auto offset = offset_x * n + offset_y;
  auto wx = v[0] - offset_x;
  auto wy = v[1] - offset_y;

  // weights
  auto w00 = (1 - wx) * (1 - wy);
  auto w01 = (1 - wx) * wy;
  auto w10 = wx * (1 - wy);
  auto w11 = wx * wy;

  auto c = prog.cache(0, 1);  // group_size = 1;
  c.store(offset);
  c.store(w00);
  c.store(w01);
  c.store(w10);
  c.store(w11);

  // change of SIMD width

  auto new_offset = c.load();
  auto address = index(0) + new_offset;

  auto load = [&](int i, int offset_x, int offset_y) {
    return prog.load(attr[i], address + offset_x * n + offset_y);
  };

  for (int i = 0; i < nattr; i++) {
    auto new_attr = w00 * load(i, 0, 0) + w01 * load(i, 0, 1) + w10 * load(i, 1, 0) +
               w11 * load(i, 1, 1);
    prog.store(new_attr, new_addr[i]);
  }

  prog();
};
TC_REGISTER_TASK(advection);
*/

TC_NAMESPACE_END