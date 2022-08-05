#include "taichi/ui/gui/gui.h"

TI_NAMESPACE_BEGIN

Vector2 Canvas::Line::vertices[128];

void Canvas::triangles_batched(int n,
                               std::size_t a_,
                               std::size_t b_,
                               std::size_t c_,
                               uint32 color_single,
                               std::size_t color_array) {
  auto a = (real *)a_;
  auto b = (real *)b_;
  auto c = (real *)c_;
  auto color_arr = (uint32 *)color_array;
  for (int i = 0; i < n; i++) {
    auto clr = color_single;
    if (color_arr) {
      clr = color_arr[i];
    }
    triangle_single(a[i * 2], a[i * 2 + 1], b[i * 2], b[i * 2 + 1], c[i * 2],
                    c[i * 2 + 1], clr);
  }
}

void Canvas::paths_batched(int n,
                           std::size_t a_,
                           std::size_t b_,
                           uint32 color_single,
                           std::size_t color_array,
                           real radius_single,
                           std::size_t radius_array) {
  auto a = (real *)a_;
  auto b = (real *)b_;
  auto color_arr = (uint32 *)color_array;
  auto radius_arr = (real *)radius_array;
  for (int i = 0; i < n; i++) {
    auto r = radius_single;
    if (radius_arr) {
      r = radius_arr[i];
    }
    auto clr = color_single;
    if (color_arr) {
      clr = color_arr[i];
    }
    // FIXME: path_single seems not displaying correct without the 1e-6 term:
    path_single(a[i * 2], a[i * 2 + 1], b[i * 2] + 1e-6 * (i % 18 + 6),
                b[i * 2 + 1], clr, r);
  }
}

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

void Canvas::path_single(real x0,
                         real y0,
                         real x1,
                         real y1,
                         uint32 color,
                         real radius) {
  path(Vector2(x0, y0), Vector2(x1, y1)).radius(radius).color(color).finish();
}

void Canvas::triangle(Vector2 a, Vector2 b, Vector2 c, Vector4 color) {
  a = transform(a);
  b = transform(b);
  c = transform(c);

  // real points[3] = {a, b, c};
  // std::sort(points, points + 3, [](const Vector2 &a, const Vector2 &b) {
  //  return a.y < b.y;
  //});
  Vector2 limits[2];
  limits[0].x = min(a.x, min(b.x, c.x));
  limits[0].y = min(a.y, min(b.y, c.y));
  limits[1].x = max(a.x, max(b.x, c.x));
  limits[1].y = max(a.y, max(b.y, c.y));
  for (int i = (int)std::floor(limits[0].x); i < (int)std::ceil(limits[1].x);
       i++) {
    for (int j = (int)std::floor(limits[0].y); j < (int)std::ceil(limits[1].y);
         j++) {
      Vector2 pixel(i + 0.5_f, j + 0.5_f);
      bool inside_a = cross(pixel - a, b - a) <= 0;
      bool inside_b = cross(pixel - b, c - b) <= 0;
      bool inside_c = cross(pixel - c, a - c) <= 0;

      // cover both clockwise and counterclockwise case for vertices [a, b, c]
      bool inside_triangle = (inside_a == inside_b) && (inside_a == inside_c);

      if (inside_triangle && img.inside(i, j)) {
        img[i][j] = color;
      }
    }
  }
}

void Canvas::triangle_single(real x0,
                             real y0,
                             real x1,
                             real y1,
                             real x2,
                             real y2,
                             uint32 color_hex) {
  auto a = Vector2(x0, y0);
  auto b = Vector2(x1, y1);
  auto c = Vector2(x2, y2);
  auto color = color_from_hex(color_hex);
  triangle(a, b, c, color);
}

TI_NAMESPACE_END
