#include <taichi/visual/gui.h>
#include <taichi/common/task.h>

TC_NAMESPACE_BEGIN

auto test_bemo = []() {
  GUI gui("Bemo Test", 1000, 400, false);
  auto canvas = *gui.canvas;
  real t = 0;

  int circle_count = 10;
  gui.button("ABC", [] { TC_INFO("Triggered"); });
  gui.slider("Circles", circle_count, 0, 60);
  // gui.button("abc", [] { TC_INFO("Triggered"); });
  // gui.button("TEST TEST TEST", [] { TC_INFO("Triggered2"); });

  while (1) {
    t += 0.02_f;
    canvas.clear(Vector4(0.95));

    for (int i = 0; i < 30; i++) {
      canvas.circle(i * 10 + 100, 250 + std::sin(t + i * 0.1_f) * 50_f)
          .color(0.7_f, 0.2_f, 0.0_f, 0.9_f)
          .radius(5);
    }
    canvas.color(0.0_f, 0.0_f, 1.0_f, 1.0_f).radius(5 + 2 * std::sin(t * 10_f));
    canvas.path()
        .path(Vector2(100, 100), Vector2(200, 75 + std::cos(t) * 50_f),
              Vector2(300, 75 + std::cos(t) * 50_f))
        .close()
        .color(0, 0, 0)
        .width(5);

    for (int i = 0; i < circle_count; i++) {
      canvas.circle(i * 10 + 100, 150 + std::sin(t + i * 0.1_f) * 50_f);
    }
    gui.update();
  }
};

TC_REGISTER_TASK(test_bemo);

TC_NAMESPACE_END
