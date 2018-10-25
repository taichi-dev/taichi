#include <taichi/visual/gui.h>
#include <taichi/common/task.h>

TC_NAMESPACE_BEGIN

auto test_bemo = []() {
  GUI gui("Bemo Test", 800, 400);
  auto canvas = *gui.canvas;
  real t = 0;

  while (1) {
    t += 0.02_f;
    canvas.clear(Vector4(0.95));

    for (int i = 0; i < 60; i++) {
      canvas.circle(i * 10 + 100, 200 + std::sin(t + i * 0.1) * 50)
          .color(0.7, 0.2, 0.0, 0.9)
          .radius(5);

    }
    canvas.line(100, 100, 200, 75 + std::cos(t) * 50)
        .color(0.0, 0.0, 0.0, 0.5)
        .width(5);
    gui.update();
  }
};

TC_REGISTER_TASK(test_bemo);

TC_NAMESPACE_END
