#include <taichi/visual/gui.h>

TC_NAMESPACE_BEGIN

Vector2 Canvas::Line::vertices[128];

void Canvas::circles_batched(int n,
                             std::size_t x_,
                             uint32 color_single,
                             std::size_t color_array,
                             real radius_single,
                             std::size_t radius_array) {
  auto x = (real *)x_;
  auto color_arr = (uint32 *)color_array;
  auto radius_arr = (real *)radius_array;
  for (int i = 0; i < n; i++) {
    auto r = radius_single;
    if (radius_arr) {
      r = radius_arr[i];
    }
    auto c = color_single;
    if (color_arr) {
      c = color_arr[i];
    }
    circle(x[i * 2], x[i * 2 + 1]).radius(r).color(c).finish();
  }
}

void Canvas::circle_single(real x, real y, uint32 color, real radius) {
  circle(x, y).radius(radius).color(color).finish();
}

TC_NAMESPACE_END
