#pragma once

#include <taichi/taichi>
#include <numeric>

TC_NAMESPACE_BEGIN

#if defined(TC_PLATFORM_LINUX) || \
    (defined(TC_PLATFORM_OSX) && defined(TC_AMALGAMATED))
#define TC_GUI_X11
#endif

#if defined(TC_PLATFORM_WINDOWS)
#define TC_GUI_WIN32
#endif

#if defined(TC_PLATFORM_OSX)
#define TC_GUI_COCOA
#endif

class Canvas {
  struct Context {
    Vector4 _color;
    real _radius;
  };

 public:
  Context context;

  Canvas &color(Vector4 val) {
    context._color = val;
    return *this;
  }

  TC_FORCE_INLINE Canvas &color(real r, real g, real b, real a = 1) {
    context._color = Vector4(r, g, b, a);
    return *this;
  }

  TC_FORCE_INLINE Canvas &color(int r, int g, int b, int a = 255) {
    context._color = (1.0_f / 255) * Vector4(r, g, b, a);
    return *this;
  }

  TC_FORCE_INLINE Canvas &radius(real radius) {
    context._radius = radius;
    return *this;
  }

  struct Line {
    const Canvas &canvas;
    Vector2 a, b;
    Vector4 _color;
    real _radius;

    TC_FORCE_INLINE Line(Canvas &canvas, Vector2 a, Vector2 b)
        : canvas(canvas),
          a(a),
          b(b),
          _color(canvas.context._color),
          _radius(canvas.context._radius) {
    }

    TC_FORCE_INLINE Line &color(Vector4 color) {
      _color = color;
      return *this;
    }

    TC_FORCE_INLINE Line &color(real r, real g, real b, real a = 1) {
      _color = Vector4(r, g, b, a);
      return *this;
    }

    TC_FORCE_INLINE Line &color(int r, int g, int b, int a = 255) {
      _color = (1.0_f / 255) * Vector4(r, g, b, a);
      return *this;
    }

    TC_FORCE_INLINE Line &width(real width) {
      _radius = width * 0.5;
      return *this;
    }

    TC_FORCE_INLINE Line &radius(real radius) {
      _radius = radius;
      return *this;
    }

    // TODO: end style e.g. arrow

    TC_FORCE_INLINE ~Line() {
      // TODO: accelerate
      auto a_i = (a + Vector2(0.5_f)).template cast<int>();
      auto b_i = (b + Vector2(0.5_f)).template cast<int>();
      auto radius_i = (int)std::ceil(_radius + 0.5_f);
      auto range_lower = Vector2i(std::min(a_i.x, b_i.x) - radius_i,
                                  std::min(a_i.y, b_i.y) - radius_i);
      auto range_higher = Vector2i(std::max(a_i.x, b_i.x) + radius_i,
                                   std::max(a_i.y, b_i.y) + radius_i);
      auto direction = normalized(b - a);
      auto l = length(b - a);
      auto tangent = Vector2(-direction.y, direction.x);
      for (int i = range_lower.x; i <= range_higher.x; i++) {
        for (int j = range_lower.y; j <= range_higher.y; j++) {
          auto pixel_coord = Vector2(i + 0.5_f, j + 0.5_f) - a;
          auto u = dot(tangent, pixel_coord);
          auto v = dot(direction, pixel_coord);
          if (v > 0) {
            v = std::max(0.0_f, v - l);
          }
          real dist = length(Vector2(u, v));
          auto alpha = _color.w * clamp(_radius - dist);
          auto &dest = canvas.img[Vector2i(i, j)];
          dest = lerp(alpha, dest, _color);
        }
      }
    }
  };

  struct Circle {
    const Canvas &canvas;
    Vector2 _center;
    Vector4 _color;
    real _radius;

    TC_FORCE_INLINE Circle(Canvas &canvas, Vector2 center)
        : canvas(canvas),
          _center(center),
          _color(canvas.context._color),
          _radius(canvas.context._radius) {
    }

    TC_FORCE_INLINE Circle &color(Vector4 color) {
      _color = color;
      return *this;
    }

    TC_FORCE_INLINE Circle &color(real r, real g, real b, real a = 1) {
      _color = Vector4(r, g, b, a);
      return *this;
    }

    TC_FORCE_INLINE Circle &color(int r, int g, int b, int a = 255) {
      _color = (1.0_f / 255) * Vector4(r, g, b, a);
      return *this;
    }

    TC_FORCE_INLINE Circle &radius(real radius) {
      _radius = radius;
      return *this;
    }

    TC_FORCE_INLINE ~Circle() {
      auto center_i = (_center + Vector2(0.5_f)).template cast<int>();
      auto radius_i = (int)std::ceil(_radius + 0.5_f);
      for (int i = -radius_i; i <= radius_i; i++) {
        for (int j = -radius_i; j <= radius_i; j++) {
          real dist =
              length(_center - center_i.template cast<real>() - Vector2(i, j));
          auto alpha = _color.w * clamp(_radius - dist);
          auto &dest = canvas.img[center_i + Vector2i(i, j)];
          dest = lerp(alpha, dest, _color);
        }
      }
    }
  };

 public:
  Array2D<Vector4> &img;
  Matrix3 transform_matrix;

  Canvas(Array2D<Vector4> &img) : img(img) {
    transform_matrix = Matrix3(Vector3(img.get_res().cast<real>(), 1.0_f));
  }

  TC_FORCE_INLINE Vector2 transform(Vector2 x) {
    return Vector2(transform_matrix * Vector3(x, 1.0_f));
  }

  Circle circle(Vector2 center) {
    return Circle(*this, center);
  }

  Circle circle(real x, real y) {
    return Circle(*this, Vector2(x, y));
  }

  Line line(real xa, real ya, real xb, real yb) {
    return line(Vector2(xa, ya), Vector2(xb, yb));
  }

  Line line(Vector2 a, Vector2 b) {
    return Line(*this, a, b);
  }

  void line(Vector2 start, Vector2 end, Vector4 color) {
    // convert to screen space
    start = transform(start);
    end = transform(end);
    real len = length(end - start);
    int samples = (int)len * 2 + 4;
    for (int i = 0; i < samples; i++) {
      real alpha = (1.0_f / (samples - 1)) * i;
      Vector2i coord = (lerp(alpha, start, end)).floor().cast<int>();
      if (img.inside(coord)) {
        img[coord] = color;
      }
    }
  }

  void triangle(Vector2 a, Vector2 b, Vector2 c, Vector4 color) {
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
      for (int j = (int)std::floor(limits[0].y);
           j < (int)std::ceil(limits[1].y); j++) {
        Vector2 pixel(i + 0.5_f, j + 0.5_f);
        bool inside_a = cross(pixel - a, b - a) <= 0;
        bool inside_b = cross(pixel - b, c - b) <= 0;
        bool inside_c = cross(pixel - c, a - c) <= 0;
        if (inside_a && inside_b && inside_c && img.inside(i, j)) {
          img[i][j] = color;
        }
      }
    }
  }

  void text(const std::string &str,
            Vector2 position,
            real size,
            Vector4 color) {
    position = transform(position);
    char *root_dir = std::getenv("TAICHI_REPO_DIR");
    TC_ASSERT(root_dir != nullptr);
    img.write_text(root_dir + std::string("/assets/fonts/go/Go-Bold.ttf"), str,
                   size, position.x, position.y, color);
  }

  void clear(Vector4 color) {
    img.reset(color);
  }

  ~Canvas() {
  }
};

#if defined(TC_GUI_X11)

class CXImage;

class GUIBaseX11 {
 public:
  void *display;
  void *visual;
  unsigned long window;
  std::unique_ptr<CXImage> img;
};

using GUIBase = GUIBaseX11;

#endif

#if defined(TC_GUI_WIN32)
class GUIBaseWin32 {
 public:
  HWND hwnd;
  HDC hdc;
  COLORREF *data;
  HDC src;
  HBITMAP bitmap;
};

using GUIBase = GUIBaseWin32;
#endif

#if defined(TC_GUI_COCOA)
class GUIBaseCocoa {
 public:
};

using GUIBase = GUIBaseCocoa;
#endif

class GUI : public GUIBase {
 public:
  std::string window_name;
  int width, height;
  int frame_id = 0;
  const int fps = 60;
  float64 start_time;
  Array2D<Vector4> buffer;
  std::vector<real> last_frame_interval;
  std::unique_ptr<Canvas> canvas;
  float64 last_frame_time;
  bool key_pressed;
  std::vector<std::string> log_entries;

  void process_event();

  GUI(const std::string &window_name, int width = 800, int height = 800);

  GUI(const std::string &window_name, Vector2i res)
      : GUI(window_name, res[0], res[1]) {
  }

  Canvas &get_canvas() {
    return *canvas;
  }

  void redraw();

  void update();

  void wait_key();

  void draw_log() {
    for (int i = 0; i < (int)log_entries.size(); i++) {
      canvas->text(log_entries[i], Vector2(0.0, -0.02_f * i), 15, Vector4(0));
    }
  }

  void log(std::string entry) {
    log_entries.push_back(entry);
    if (log_entries.size() > 15) {
      log_entries.erase(log_entries.begin());
    }
  }

  ~GUI();
};

TC_NAMESPACE_END
