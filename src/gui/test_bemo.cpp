#include <taichi/visual/gui.h>
#include <taichi/common/task.h>

TC_NAMESPACE_BEGIN

auto test_bemo = []() {
  GUI gui("Bemo Test", 800, 400);
  auto canvas = *gui.canvas;
  real t = 0;
  while (1) {
    t += 0.02_f;
    canvas.clear(Vector4(1, 1, 1, 1));

    for (int i = 0; i < 60; i++) {
      canvas.circle(i * 10 + 100, 200 + std::sin(t + i * 0.1) * 50).color(0.3, 0.2, 0.0, 0.5).radius(5);
    }
    gui.update();
  }
};

TC_REGISTER_TASK(test_bemo);

TC_NAMESPACE_END
