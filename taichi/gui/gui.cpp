#include <taichi/visual/gui.h>

TC_NAMESPACE_BEGIN

Vector2 Canvas::Line::vertices[128];

void Canvas::circles_batched(int n,
                                     std::size_t x_,
                                     std::size_t color_,
                                     real radius) {
  auto x = (real *)x_;
  auto color = (uint32 *)color_;
  for (int i = 0; i < n; i++) {
    circle(x[i * 2], x[i * 2 + 1]).radius(radius).color(color[i]).finish();
  }
}

TC_NAMESPACE_END
