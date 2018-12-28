#include "tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/common/bit.h>

TC_NAMESPACE_BEGIN

using namespace Tlang;

auto mpm = []() {
  bool use_adapter = true;

  const int n = 1024, nattr = 4;

  Float attr[2][nattr], v[2];

  Program prog(Arch::x86_64, n * n);

  prog.config.group_size = use_adapter ? nattr : 1;
  prog.config.num_groups = use_adapter ? 8 : 8;

  prog.layout([&]() {
    for (int k = 0; k < 2; k++) {
      if (use_adapter) {
        for (int i = 0; i < nattr; i++) {
          prog.buffer(k).range(n * n).stream(0).group(0).place(attr[k][i]);
        }
        prog.buffer(2).range(n * n).stream(0).group(0).place(v[k]);
      } else {
        for (int i = 0; i < nattr; i++) {
          prog.buffer(k).range(n * n).stream(i).group(0).place(attr[k][i]);
        }
        prog.buffer(2).range(n * n).stream(k).group(0).place(v[k]);
      }
    }
  });

  TC_ASSERT(bit::is_power_of_two(n));

  auto func = prog.def([&]() {
    auto index = Expr::index(0);
    for_loop(index, {0, n * n}, [&] {
      // ** gs = 2

      auto offset_x = floor(v[0][index]).name("offset_x");
      auto offset_y = floor(v[1][index]).name("offset_y");
      auto wx = v[0][index] - offset_x;
      auto wy = v[1][index] - offset_y;
      wx.name("wx");
      wy.name("wy");

      if (use_adapter) {
        prog.adapter(0).set(2, 1).convert(offset_x, offset_y);
        prog.adapter(1).set(2, 1).convert(wx, wy);
      }

      // ** gs = 1
      auto offset =
          cast<int32>(offset_x) * imm(n) + cast<int32>(offset_y) * imm(1);

      auto clamp = [](const Expr &e) { return min(max(imm(2), e), imm(n - 2)); };

      // weights
      auto w00 = (imm(1.0f) - wx) * (imm(1.0f) - wy);
      auto w01 = (imm(1.0f) - wx) * wy;
      auto w10 = wx * (imm(1.0f) - wy);
      auto w11 = wx * wy;

      w00.name("w00");
      w01.name("w01");
      w10.name("w10");
      w11.name("w11");

      Expr node = max(Expr::index(0) + offset, imm(0));
      Int32 i = clamp(node >> imm((int)bit::log2int(n))).name("i");  // node / n
      // Int32 i = clamp(node / imm(n)).name("i"); // node / n
      Int32 j = clamp(node & imm(n - 1)).name("j");  // node % n
      // Int32 j = clamp(node % imm(n)).name("j"); // node % n
      node = i * imm(n) + j;
      node.name("node");

      if (use_adapter) {
        prog.adapter(2).set(1, 4).convert(w00, w01, w10, w11);
        prog.adapter(3).set(1, 4).convert(node);
      }

      // ** gs = 4
      for (int k = 0; k < nattr; k++) {
        auto v00 = attr[0][k][node + imm(0)].name("v00");
        auto v01 = attr[0][k][node + imm(1)].name("v01");
        auto v10 = attr[0][k][node + imm(n)].name("v10");
        auto v11 = attr[0][k][node + imm(n + 1)].name("v11");

        attr[1][k][index] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
        // attr[1][k][index] = w00 * v00;
        attr[1][k][index].name(fmt::format("output{}", k));
      }
    });
  });


  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < nattr; k++) {
        prog.data(attr[0][k], i * n + j) = i % 128 / 128.0_f;
      }
      real s = 20.0_f / n;
      prog.data(v[0], i * n + j) = s * (j - n / 2);
      prog.data(v[1], i * n + j) = -s * (i - n / 2);
      // prog.data(v[0], i * n + j) = 0;
      // prog.data(v[1], i * n + j) = 0;
    }
  }

  GUI gui("Advection", n, n);

  for (int f = 0; f < 1000; f++) {
    for (int t = 0; t < 3; t++) {
      TC_TIME(func());

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < nattr; k++) {
            gui.buffer[i][j] = Vector4(prog.data(attr[1][k], i * n + j));
          }
        }
      }

      prog.swap_buffers(0, 1);
    }

    gui.update();
    // gui.screenshot(fmt::format("images/{:04d}.png", f));
  }
};
TC_REGISTER_TASK(mpm);

TC_NAMESPACE_END

/*
TODO: arbitrary for loop (bounds using arbitrary constants)
 */
