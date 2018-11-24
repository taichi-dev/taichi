#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/visual/gui.h>

TC_NAMESPACE_BEGIN

constexpr int n = 100;
constexpr real dx = 1.0_f / n;
constexpr real c = 340;
constexpr real alpha = 0.001;

Array2D<real> p, q, r;

void advance(real dt) {
  r.reset_zero();
  constexpr real inv_dx2 = pow<2>(1.0_f / dx);
  for (int i = 1; i < n - 1; i++) {
    for (int j = 1; j < n - 1; j++) {
      real laplacian_p = inv_dx2 * (p[i - 1][j] + p[i][j - 1] + p[i + 1][j] +
                                    p[i][j + 1] - 4 * p[i][j]);
      real laplacian_q = inv_dx2 * (q[i - 1][j] + q[i][j - 1] + q[i + 1][j] +
                                    q[i][j + 1] - 4 * q[i][j]);

      r[i][j] = 2 * q[i][j] + (c * c * dt * dt + c * alpha * dt) * laplacian_q -
                p[i][j] - c * alpha * dt * laplacian_p;
    }
  }
  std::swap(p.data, q.data);
  std::swap(q.data, r.data);
}

auto sound = []() {
  int window_size = 800;
  int scale = window_size / n;
  q.initialize(Vector2i(n));
  p = r = q;
  GUI gui("Sound simulation", Vector2i(window_size));
  real t = 0, dt = (std::sqrt(alpha * alpha + dx * dx / 3) - alpha) / c;
  // p[n / 2][n / 2] = std::sin(t);
  p[n / 2][n / 2] = 1;
  while (1) {
    for (int i = 0; i < 10; i++) {
      advance(dt);
    }
    t += dt;
    for (int i = 0; i < window_size; i++) {
      for (int j = 0; j < window_size; j++) {
        auto c = p[i / scale][j / scale];
        gui.get_canvas().img[i][j] = Vector3(c + 0.5_f);
      }
    }
    gui.update();
  }
};

TC_REGISTER_TASK(sound);

TC_NAMESPACE_END
