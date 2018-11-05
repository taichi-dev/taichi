#pragma once

#include <taichi/math.h>
#include <taichi/system/timer.h>
#include <numeric>

#if defined(TC_PLATFORM_LINUX)
#define TC_GUI_X11
#endif

#if defined(TC_PLATFORM_WINDOWS)
#define TC_GUI_WIN32
#endif

#if defined(TC_PLATFORM_OSX)
#define TC_GUI_COCOA
#include <objc/objc.h>
#endif

TC_NAMESPACE_BEGIN

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
    Canvas &canvas;
    Vector4 _color;
    real _radius;
    int n_vertices;
    static Vector2 vertices[128];  // TODO: ...

    TC_FORCE_INLINE Line(Canvas &canvas)
        : canvas(canvas),
          _color(canvas.context._color),
          _radius(canvas.context._radius) {
      n_vertices = 0;
    }

    TC_FORCE_INLINE Line(Canvas &canvas, Vector2 a, Vector2 b) : Line(canvas) {
      push(a);
      push(b);
    }

    TC_FORCE_INLINE Line(Canvas &canvas, Vector2 a, Vector2 b, Vector2 c)
        : Line(canvas) {
      push(a);
      push(b);
      push(c);
    }

    TC_FORCE_INLINE Line(Canvas &canvas,
                         Vector2 a,
                         Vector2 b,
                         Vector2 c,
                         Vector2 d)
        : Line(canvas) {
      push(a);
      push(b);
      push(c);
      push(d);
    }

    TC_FORCE_INLINE void push(Vector2 vec) {
      vertices[n_vertices++] = vec;
    }

    TC_FORCE_INLINE Line &path(Vector2 a) {
      push(a);
      return *this;
    }

    TC_FORCE_INLINE Line &path(Vector2 a, Vector2 b) {
      push(a);
      push(b);
      return *this;
    }

    TC_FORCE_INLINE Line &path(Vector2 a, Vector2 b, Vector2 c) {
      push(a);
      push(b);
      push(c);
      return *this;
    }

    TC_FORCE_INLINE Line &path(Vector2 a, Vector2 b, Vector2 c, Vector2 d) {
      push(a);
      push(b);
      push(c);
      push(d);
      return *this;
    }

    TC_FORCE_INLINE Line &close() {
      TC_ASSERT(n_vertices > 0);
      push(vertices[0]);
      return *this;
    }

    TC_FORCE_INLINE Line &color(Vector4 color) {
      _color = color;
      return *this;
    }

    TC_FORCE_INLINE Line &color(int c) {
      return color(c / 65536, c / 256 % 256, c % 256, 255);
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

    void stroke(Vector2 a, Vector2 b) {
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

    TC_FORCE_INLINE ~Line() {
      for (int i = 0; i + 1 < n_vertices; i++) {
        stroke(canvas.transform(vertices[i]),
               canvas.transform(vertices[i + 1]));
      }
    }
  };

  struct Circle {
    Canvas &canvas;
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

    TC_FORCE_INLINE Circle &color(int c) {
      return color(c / 65536, c / 256 % 256, c % 256, 255);
    }

    TC_FORCE_INLINE Circle &radius(real radius) {
      _radius = radius;
      return *this;
    }

    TC_FORCE_INLINE ~Circle() {
      auto center = canvas.transform(_center);
      auto center_i = (center + Vector2(0.5_f)).template cast<int>();
      auto radius_i = (int)std::ceil(_radius + 0.5_f);
      for (int i = -radius_i; i <= radius_i; i++) {
        for (int j = -radius_i; j <= radius_i; j++) {
          real dist =
              length(center - center_i.template cast<real>() - Vector2(i, j));
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

  TC_FORCE_INLINE Vector2 transform(Vector2 x) const {
    return Vector2(transform_matrix * Vector3(x, 1.0_f));
  }

  Circle circle(Vector2 center) {
    return Circle(*this, center);
  }

  Circle circle(real x, real y) {
    return Circle(*this, Vector2(x, y));
  }

  Line path(real xa, real ya, real xb, real yb) {
    return path(Vector2(xa, ya), Vector2(xb, yb));
  }

  Line path() {
    return Line(*this);
  }

  Line path(Vector2 a, Vector2 b) {
    return Line(*this).path(a, b);
  }

  Line path(Vector2 a, Vector2 b, Vector2 c) {
    return Line(*this, a, b, c);
  }

  Line path(Vector2 a, Vector2 b, Vector2 c, Vector2 d) {
    return Line(*this, a, b, c, d);
  }

  Line rect(Vector2 a, Vector2 b) {
    return Line(*this, a, Vector2(a.x, b.y), b, Vector2(b.x, a.y));
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
    img.write_text(root_dir + std::string("/assets/fonts/go/Go-Regular.ttf"),
                   str, size, position.x, position.y, color);
  }

  void clear(Vector4 color) {
    img.reset(color);
  }

  void clear(int c) {
    img.reset((1.0_f / 255) * Vector4(c / 65536, c / 256 % 256, c % 256, 255));
  }

  ~Canvas() {
  }

  void set_idendity_transform_matrix() {
    transform_matrix = Matrix3(1);
  }
};

#if defined(TC_GUI_X11)

class CXImage;

class GUIBaseX11 {
 public:
  void *display;
  void *visual;
  unsigned long window;
  CXImage *img;
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
  id window, view;
  std::size_t img_data_length;
  std::vector<uint8_t> img_data;
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
  Vector2i cursor_pos;

  void set_mouse_pos(int x, int y) {
    cursor_pos = Vector2i(x, y);
  }

  Vector2i widget_size = Vector2i(200, 40);

  struct MouseEvent {
    enum class Type { move, press, release };
    Type type;
    Vector2i pos;
  };

  struct Rect {
    Vector2i pos;
    Vector2i size;
    TC_IO_DEF(pos, size);
    Rect() {
    }
    Rect(Vector2i pos, Vector2i size) : pos(pos), size(size) {
    }
    bool inside(Vector2i p) {
      return pos <= p && p < pos + size;
    }
  };

  class Widget {
   public:
    Rect rect;
    Widget(Rect rect) : rect(rect){};
    bool hover;

    Widget() {
      hover = false;
    }

    bool inside(Vector2i p) {
      return rect.inside(p);
    }

    virtual void mouse_event(MouseEvent e) {
    }

    virtual void redraw(Canvas &canvas) {
      /*
      canvas
          .rect(rect.pos.template cast<real>(),
                (rect.pos + rect.size).template cast<real>())
          .color(Vector4(1.0, 0.0, 0.0, 1))
          .close()
          .radius(3);
      */
      for (int i = 1; i < rect.size[0] - 1; i++) {
        for (int j = 1; j < rect.size[1] - 1; j++) {
          canvas.img[rect.pos[0] + i][rect.pos[1] + j] =
              Vector4(0.5, 0.5 + 0.3 * real(hover), 0.5, 1);
        }
      }
    }

    void set_hover(bool val) {
      hover = val;
    }
  };

  std::vector<std::unique_ptr<Widget>> widgets;

  class Button : public Widget {
   public:
    std::string text;

    using CallbackType = std::function<void()>;
    CallbackType callback;

    Button(Rect rect, const std::string text, const CallbackType &callback)
        : Widget(rect), text(text), callback(callback) {
    }

    void mouse_event(MouseEvent e) override {
      if (e.type == MouseEvent::Type::release) {
        callback();
      }
    }

    void redraw(Canvas &canvas) override {
      Widget::redraw(canvas);
      int s = 32;
      canvas.text(
          text,
          (rect.pos + Vector2i(2, rect.size[1] - 2)).template cast<real>(), s,
          Vector4f(0));
    }
  };

  template <typename T>
  class Slider : public Widget {
   public:
    std::string text;
    T &val;
    T minimum, maximum, step;

    using CallbackType = std::function<void()>;
    CallbackType callback;

    Slider(Rect rect,
           const std::string text,
           T &val,
           T minimum,
           T maximum,
           T step)
        : Widget(rect),
          text(text),
          val(val),
          minimum(minimum),
          maximum(maximum),
          step(step) {
    }

    void mouse_event(MouseEvent e) override {
      if (e.type == MouseEvent::Type::press) {
        // Compute val
        T new_val = minimum;
        val = new_val;
      }
    }

    void redraw(Canvas &canvas) override {
      Widget::redraw(canvas);
      int s = 16;
      canvas.text(
          text,
          (rect.pos + Vector2i(2, rect.size[1] - 2)).template cast<real>(), s,
          Vector4f(0));
      auto slider_padding = 5;
      int slider_start = slider_padding,
          slider_end = rect.size[0] - slider_padding;
      for (int i = slider_start; i < slider_end; i++) {
        for (int j = slider_padding; j < slider_padding + 3; j++) {
          canvas.img[rect.pos[0] + i][rect.pos[1] + j] = Vector4(1, 0, 0, 1);
        }
      }
      static int t = 0;
      auto alpha = 0.5 * std::sin((t++) * 0.01) + 0.5;
      canvas
          .circle(rect.pos.template cast<real>() +
                  Vector2(lerp(alpha, slider_start, slider_end),
                          slider_padding + 1))
          .radius(5)
          .color(Vector4(0, 1, 0, 0.8));
    }
  };

  Rect make_widget_rect() {
    return Rect(Vector2i(width - widget_size[0],
                         height - (widgets.size() + 1) * widget_size[1]) -
                    Vector2i(5),
                widget_size);
  }

  GUI &button(std::string text, const Button::CallbackType &callback) {
    widgets.push_back(
        std::make_unique<Button>(make_widget_rect(), text, callback));
    return *this;
  }

  template <typename T>
  GUI &slider(std::string text, T &val, T minimum, T maximum, T step = 1) {
    widgets.push_back(std::make_unique<Slider<T>>(make_widget_rect(), text, val,
                                                  minimum, maximum, step));
    return *this;
  }

  void process_event();

  void mouse_event(MouseEvent e) {
    for (auto &w : widgets) {
      if (w->inside(e.pos)) {
        w->mouse_event(e);
      }
    }
  }

  explicit GUI(const std::string &window_name,
               int width = 800,
               int height = 800,
               bool normalized_coord = true)
      : window_name(window_name),
        width(width),
        height(height),
        key_pressed(false) {
    create_window();
    set_title(window_name);
    start_time = taichi::Time::get_time();
    buffer.initialize(Vector2i(width, height));
    canvas = std::make_unique<Canvas>(buffer);
    last_frame_time = taichi::Time::get_time();
    if (!normalized_coord) {
      canvas->set_idendity_transform_matrix();
    }
  }

  explicit GUI(const std::string &window_name,
               Vector2i res,
               bool normalized_coord = true)
      : GUI(window_name, res[0], res[1], normalized_coord) {
  }

  void create_window();

  Canvas &get_canvas() {
    return *canvas;
  }

  void redraw();

  void set_title(std::string title);

  void redraw_widgets() {
    for (auto &w : widgets) {
      w->set_hover(w->inside(cursor_pos));
      w->redraw(*canvas);
    }
  }

  void update() {
    frame_id++;
    redraw_widgets();
    while (taichi::Time::get_time() < start_time + frame_id / (real)fps)
      ;
    redraw();
    process_event();
    while (last_frame_interval.size() > 30) {
      last_frame_interval.erase(last_frame_interval.begin());
    }
    auto real_fps = last_frame_interval.size() /
                    (std::accumulate(last_frame_interval.begin(),
                                     last_frame_interval.end(), 0.0_f));
    set_title(fmt::format("{} ({:.02f} FPS)", window_name, real_fps));
    if (last_frame_time != 0) {
      last_frame_interval.push_back(taichi::Time::get_time() - last_frame_time);
    }
    last_frame_time = taichi::Time::get_time();
  }

  void wait_key() {
    while (true) {
      key_pressed = false;
      update();
      if (key_pressed) {
        break;
      }
    }
  }

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
