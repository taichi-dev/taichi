auto advection = []() {
  bool use_adapter = false;

  const int dim = 2;

  const int n = 1024, nattr = 4;
  const int block_size = 16;
  TC_ASSERT(n % block_size == 0);
  auto x = ind(), y = ind();

  Float attr[dim][nattr], v[dim];

  Program prog(Arch::x86_64);

  prog.config.group_size = use_adapter ? nattr : 1;
  prog.config.num_groups = use_adapter ? 8 : 8;

  layout([&]() {
    for (int k = 0; k < dim; k++) {
      if (use_adapter) {
        TC_NOT_IMPLEMENTED
        for (int i = 0; i < nattr; i++) {
          // prog.buffer(k).range(n * n).stream(0).group(0).place(attr[k][i]);
        }
        // prog.buffer(2).range(n * n).stream(0).group(0).place(v[k]);
      } else {
        if (block_size > 1) {
          for (int i = 0; i < nattr; i++) {
            attr[k][i] = var<float32>();
            root.fixed({x, y}, {n / block_size, n / block_size})
                .fixed({x, y}, {block_size, block_size})
                .place(attr[k][i]);
          }
          v[k] = var<float32>();
          root.fixed({x, y}, {n / block_size, n / block_size})
              .fixed({x, y}, {block_size, block_size})
              .place(v[k]);
        } else {
          for (int i = 0; i < nattr; i++) {
            attr[k][i] = var<float32>();
            root.fixed({x, y}, {n, n}).place(attr[k][i]);
          }
          v[k] = var<float32>();
          root.fixed({x, y}, {n, n}).place(v[k]);
        }
      }
    }
  });

  TC_ASSERT(bit::is_power_of_two(n));

  auto clamp = [](const Expr &e) { return min(max(imm(0.0_f), e), imm(n - 2.0_f)); };

  auto func = kernel(attr[0][0], [&]() {
    // ** gs = 2

    auto vx = v[0][x, y];
    auto vy = v[1][x, y];

    if (use_adapter) {
      // prog.adapter(0).set(2, 1).convert(offset_x, offset_y);
      // prog.adapter(1).set(2, 1).convert(wx, wy);
    }

    // ** gs = 1

    auto new_x_f = clamp(vx + cast<float32>(x));
    auto new_y_f = clamp(vy + cast<float32>(y));
    auto new_x = cast<int32>(floor(new_x_f));
    auto new_y = cast<int32>(floor(new_y_f));

    auto wx = new_x_f - cast<float32>(new_x);
    auto wy = new_y_f - cast<float32>(new_y);

    // weights
    auto w00 = (imm(1.0f) - wx) * (imm(1.0f) - wy);
    auto w01 = (imm(1.0f) - wx) * wy;
    auto w10 = wx * (imm(1.0f) - wy);
    auto w11 = wx * wy;

    if (use_adapter) {
      prog.adapter(2).set(1, 4).convert(w00, w01, w10, w11);
      // prog.adapter(3).set(1, 4).convert(node);
    }

    // ** gs = 4
    for (int k = 0; k < nattr; k++) {
      auto v00 = attr[0][k][new_x, new_y].name("v00");
      auto v01 = attr[0][k][new_x, new_y + imm(1)].name("v01");
      auto v10 = attr[0][k][new_x + imm(1), new_y].name("v10");
      auto v11 = attr[0][k][new_x + imm(1), new_y + imm(1)].name("v11");
      attr[1][k][x, y] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
      attr[1][k][x, y].name(fmt::format("output{}", k));
    }
  });

  auto swap_buffers = kernel(attr[0][0], [&] {
    for (int i = 0; i < nattr; i++) {
      attr[0][i][x, y] = attr[1][i][x, y];
    }
  });

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < nattr; k++) {
        attr[0][k].val<float32>(i, j) = i % 128 / 128.0_f;
      }
      real s = 20.0_f / n;
      v[0].val<float32>(i, j) = s * (j - n / 2);
      v[1].val<float32>(i, j) = -s * (i - n / 2);
    }
  }

  GUI gui("Advection", n, n);

  for (int f = 0; f < 1000; f++) {
    for (int t = 0; t < 10; t++) {
      TC_TIME(func());
      TC_TIME(swap_buffers());
    }

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < nattr; k++) {
          gui.buffer[i][j] = Vector4(attr[1][k].val<float32>(i, j));
        }
      }
    }
    gui.update();
    // gui.screenshot(fmt::format("images/{:04d}.png", f));
  }
};
TC_REGISTER_TASK(advection);
