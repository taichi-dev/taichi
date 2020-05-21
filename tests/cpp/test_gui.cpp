// Note: this is not really a test case.

#include "taichi/gui/gui.h"
#include "taichi/common/task.h"

TI_NAMESPACE_BEGIN

auto test_gui = []() {
  GUI gui("GUI Test", 1000, 400, false);
  auto canvas = *gui.canvas;
  real t = 0;

  int circle_count = 10;
  gui.button("ABC", [] { TI_INFO("Triggered"); });
  gui.slider("Circles", circle_count, 0, 60);
  real radius = 3;
  gui.slider("Radius", radius, 0.0_f, 10.0_f);

  while (1) {
    t += 0.02_f;
    canvas.clear(Vector4(0.95));

    for (int i = 0; i < 30; i++) {
      canvas.circle(i * 10 + 100, 250 + std::sin(t + i * 0.1_f) * 50_f)
          .color(0.7_f, 0.2_f, 0.0_f, 0.9_f)
          .radius(5)
          .finish();
    }
    canvas.color(0.0_f, 0.0_f, 1.0_f, 1.0_f).radius(5 + 2 * std::sin(t * 10_f));
    canvas.path()
        .path(Vector2(100, 100), Vector2(200, 75 + std::cos(t) * 50_f),
              Vector2(300, 75 + std::cos(t) * 50_f))
        .close()
        .color(0, 0, 0)
        .width(5)
        .finish();

    for (int i = 0; i < circle_count; i++) {
      canvas.circle(i * 10 + 100, 150 + std::sin(t + i * 0.1_f) * 50_f)
          .radius(radius)
          .finish();
    }
    gui.update();
  }
};

TI_REGISTER_TASK(test_gui);

TI_NAMESPACE_END
