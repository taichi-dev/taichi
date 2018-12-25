auto advection = []() {
  const int n = 1024, nattr = 4;

  // attr[0] are input attributes to advect, which after advection are stored in attr[1]
  // v is velocity
  Float attr[2][nattr], v[2];

  Program prog(Arch::x86_64, n * n);

  for (int k = 0; k < 2; k++) {
    if (use_adapter) {
      for (int i = 0; i < nattr; i++) {
        prog.buffer(k).stream(0).group(0).place(attr[k][i]);
      }
      prog.buffer(2).stream(0).group(0).place(v[k]);
    } else {
      for (int i = 0; i < nattr; i++) {
        prog.buffer(k).stream(i).group(0).place(attr[k][i]);
      }
      prog.buffer(2).stream(k).group(0).place(v[k]);
    }
  }
  prog.config.group_size = use_adapter ? nattr : 1;
  prog.config.num_groups = use_adapter ? 8 : 8;

  // ***************************************************************************
  // Compute part starts
  auto index = Expr::index(0);

  auto offset_x = floor(v[0][index]).name("offset_x");
  auto offset_y = floor(v[1][index]).name("offset_y");
  auto wx = v[0][index] - offset_x;
  auto wy = v[1][index] - offset_y;

  auto offset = cast<int32>(offset_x) * imm(n) + cast<int32>(offset_y) * imm(1);

  auto clamp = [](const Expr &e) { return min(max(imm(2), e), imm(n - 2)); };

  // weights
  auto w00 = (imm(1.0f) - wx) * (imm(1.0f) - wy);
  auto w01 = (imm(1.0f) - wx) * wy;
  auto w10 = wx * (imm(1.0f) - wy);
  auto w11 = wx * wy;

  // Make sure no illegal memory load by clamping coordinates.
  // (Technically this should happen when computing "node",
  //    but let's keep it as-is to keep track of performance changes.)
  Expr node = max(Expr::index(0) + offset, imm(0));
  Int32 i = clamp(node / imm(n)); // node / n
  Int32 j = clamp(node % imm(n)); // node % n
  node = i * imm(n) + j;

  for (int k = 0; k < nattr; k++) {
    auto v00 = attr[0][k][node + imm(0)];
    auto v01 = attr[0][k][node + imm(1)];
    auto v10 = attr[0][k][node + imm(n)];
    auto v11 = attr[0][k][node + imm(n + 1)];

    attr[1][k][index] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
  }
  // Compute part ends
  // ***************************************************************************

  prog.compile();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < nattr; k++) {
        prog.data(attr[0][k], i * n + j) = i % 128 / 128.0_f;
      }
      real s = 20.0_f / n;
      prog.data(v[0], i * n + j) = s * (j - n / 2);
      prog.data(v[1], i * n + j) = -s * (i - n / 2);
    }
  }

  GUI gui("Advection", n, n);
  for (int f = 0; f < 1000; f++) {
    for (int t = 0; t < 3; t++) {
      TC_TIME(prog());
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < nattr; k++) {
            gui.buffer[i][j] = Vector4(prog.data(attr[1][k], i * n + j));
          }
        }
      }
    }
    gui.update();
    prog.swap_buffers(0, 1);
  }
};
