#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/visual/gui.h>
#include "sound.h"

TC_NAMESPACE_BEGIN

constexpr int n = 30;
constexpr real room_size = 10.0_f;
constexpr real dx = room_size / n;
constexpr real c = 340;
constexpr real alpha = 0.001;

Array3D<real> p, q, r;

void advance(real dt) {
  std::swap(p.data, q.data);
  std::swap(q.data, r.data);
  r.reset_zero();
  constexpr real inv_dx2 = pow<2>(1.0_f / dx);
  for (int i = 1; i < n - 1; i++) {
    for (int j = 1; j < n - 1; j++) {
      for (int k = 1; k < n - 1; k++) {
        real laplacian_p =
            inv_dx2 *
            (p[i - 1][j][k] + p[i][j - 1][k] + p[i + 1][j][k] + p[i][j + 1][k] +
             p[i][j][k + 1] + p[i][j][k - 1] - 6 * p[i][j][k]);
        real laplacian_q =
            inv_dx2 *
            (q[i - 1][j][k] + q[i][j - 1][k] + q[i + 1][j][k] + q[i][j + 1][k] +
             q[i][j][k + 1] + q[i][j][k - 1] - 6 * q[i][j][k]);

        r[i][j][k] = 2 * q[i][j][k] +
                     (c * c * dt * dt + c * alpha * dt) * laplacian_q -
                     p[i][j][k] - c * alpha * dt * laplacian_p;
      }
    }
  }
}

auto sound = []() {
  int window_size = 800;
  int scale = window_size / n;
  q.initialize(Vector3i(n));
  p = r = q;
  GUI gui("Sound simulation", Vector2i(window_size));
  real t = 0, dt = (std::sqrt(alpha * alpha + dx * dx / 3) - alpha) / c;
  // p[n / 2][n / 2] = std::sin(t);
  FILE *f = fopen("data/wave.txt", "r");
  WaveFile wav_file("output.wav");
  dt = 1_f / 44100;
  TC_P(dt);

  r[n / 2][n / 2][n / 2] = 1;

  for (int T = 0; T < 1000; T++) {
    for (int i = 0; i < 5; i++) {
      real left, right;
      fscanf(f, "%f%f", &left, &right);
      advance(dt);
      // r[n / 4][n / 2] = (left + right) / 65536.0;
      t += dt;
      wav_file.add_sound(dt, r[n / 4 + 4][n / 2][n / 2]);
      // wav_file.add_sound(dt, std::sin(t * 10000));
    }
    for (int i = 0; i < window_size; i++) {
      for (int j = 0; j < window_size; j++) {
        auto c = p[i / scale][j / scale][n / 2];
        gui.get_canvas().img[i][j] = Vector3(c + 0.5_f);
      }
    }
    gui.update();
    if (T % 100 == 0)
      wav_file.flush();
  }
};

TC_REGISTER_TASK(sound);

TC_NAMESPACE_END
