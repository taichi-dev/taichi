#pragma once

#include "taichi/math/math.h"
#include "taichi/system/timer.h"
#include "taichi/program/kernel_profiler.h"

#include <atomic>
#include <ctime>
#include <numeric>
#include <unordered_map>

#if defined(TI_PLATFORM_LINUX) || \
    (defined(TI_PLATFORM_UNIX) && !defined(TI_PLATFORM_OSX))
#if defined(TI_PLATFORM_ANDROID)
#define TI_GUI_ANDROID
#else
#define TI_GUI_X11
#endif
#endif

#if defined(TI_PLATFORM_WINDOWS)
#define TI_GUI_WIN32
#endif

#if defined(TI_PLATFORM_OSX)
#define TI_GUI_COCOA
#include <objc/objc.h>
#endif

TI_NAMESPACE_BEGIN

TI_FORCE_INLINE Vector4 color_from_hex(uint32 c) {
  return Vector4(c / 65536, c / 256 % 256, c % 256, 255) * (1 / 255.0_f);
}

#if (false)
constexpr uint32 text_color = 0x02547D;
constexpr uint32 widget_bg = 0x02BEC4;
constexpr uint32 widget_hover = 0xA9E8DC;
constexpr uint32 slider_bar_color = text_color;
constexpr uint32 slider_circle_color = 0x0284A8;
#else
constexpr uint32 text_color = 0x111111;
constexpr uint32 widget_bg = 0xAAAAAA;
constexpr uint32 widget_hover = 0xCCCCCC;
constexpr uint32 slider_bar_color = 0x333333;
constexpr uint32 slider_circle_color = 0x555555;
#endif

class TI_DLL_EXPORT Canvas {
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

  TI_FORCE_INLINE Canvas &color(real r, real g, real b, real a = 1) {
    context._color = Vector4(r, g, b, a);
    return *this;
  }

  TI_FORCE_INLINE Canvas &color(int r, int g, int b, int a = 255) {
    context._color = (1.0_f / 255) * Vector4(r, g, b, a);
    return *this;
  }

  TI_FORCE_INLINE Canvas &radius(real radius) {
    context._radius = radius;
    return *this;
  }

  struct Line {
    Canvas &canvas;
    Vector4 _color;
    real _radius;
    int n_vertices;
    bool finished;
    static Vector2 vertices[128];  // TODO: ...

    TI_FORCE_INLINE Line(Canvas &canvas)
        : canvas(canvas),
          _color(canvas.context._color),
          _radius(canvas.context._radius) {
      n_vertices = 0;
      finished = false;
    }

    TI_FORCE_INLINE Line(Canvas &canvas, Vector2 a, Vector2 b) : Line(canvas) {
      push(a);
      push(b);
    }

    TI_FORCE_INLINE Line(Canvas &canvas, Vector2 a, Vector2 b, Vector2 c)
        : Line(canvas) {
      push(a);
      push(b);
      push(c);
    }

    TI_FORCE_INLINE Line(Canvas &canvas,
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

    TI_FORCE_INLINE void push(Vector2 vec) {
      vertices[n_vertices++] = vec;
    }

    TI_FORCE_INLINE Line &path(Vector2 a) {
      push(a);
      return *this;
    }

    TI_FORCE_INLINE Line &path(Vector2 a, Vector2 b) {
      push(a);
      push(b);
      return *this;
    }

    TI_FORCE_INLINE Line &path(Vector2 a, Vector2 b, Vector2 c) {
      push(a);
      push(b);
      push(c);
      return *this;
    }

    TI_FORCE_INLINE Line &path(Vector2 a, Vector2 b, Vector2 c, Vector2 d) {
      push(a);
      push(b);
      push(c);
      push(d);
      return *this;
    }

    TI_FORCE_INLINE Line &close() {
      TI_ASSERT(n_vertices > 0);
      push(vertices[0]);
      return *this;
    }

    TI_FORCE_INLINE Line &color(Vector4 color) {
      _color = color;
      return *this;
    }

    TI_FORCE_INLINE Line &color(int c) {
      return color(c / 65536, c / 256 % 256, c % 256, 255);
    }

    TI_FORCE_INLINE Line &color(real r, real g, real b, real a = 1) {
      _color = Vector4(r, g, b, a);
      return *this;
    }

    TI_FORCE_INLINE Line &color(int r, int g, int b, int a = 255) {
      _color = (1.0_f / 255) * Vector4(r, g, b, a);
      return *this;
    }

    TI_FORCE_INLINE Line &width(real width) {
      _radius = width * 0.5;
      return *this;
    }

    TI_FORCE_INLINE Line &radius(real radius) {
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
      range_lower(0) = std::max(0, range_lower(0));
      range_lower(1) = std::max(0, range_lower(1));
      auto range_higher = Vector2i(std::max(a_i.x, b_i.x) + radius_i,
                                   std::max(a_i.y, b_i.y) + radius_i);
      range_higher(0) = std::min(canvas.img.get_width() - 1, range_higher(0));
      range_higher(1) = std::min(canvas.img.get_height() - 1, range_higher(1));
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

    void finish() {
      TI_ASSERT(!finished);
      finished = true;
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
    bool finished;

    TI_FORCE_INLINE Circle(Canvas &canvas, Vector2 center)
        : canvas(canvas),
          _center(center),
          _color(canvas.context._color),
          _radius(canvas.context._radius) {
      finished = false;
    }

    TI_FORCE_INLINE Circle &color(Vector4 color) {
      _color = color;
      return *this;
    }

    TI_FORCE_INLINE Circle &color(real r, real g, real b, real a = 1) {
      _color = Vector4(r, g, b, a);
      return *this;
    }

    TI_FORCE_INLINE Circle &color(int r, int g, int b, int a = 255) {
      _color = (1.0_f / 255) * Vector4(r, g, b, a);
      return *this;
    }

    TI_FORCE_INLINE Circle &color(int c) {
      return color(c / 65536, c / 256 % 256, c % 256, 255);
    }

    TI_FORCE_INLINE Circle &radius(real radius) {
      _radius = radius;
      return *this;
    }

    void finish() {
      TI_ASSERT(finished == false);
      finished = true;
      auto center = canvas.transform(_center);
      auto const canvas_width = canvas.img.get_width();
      auto const canvas_height = canvas.img.get_height();
      const auto r = _radius;
      int i_lower = std::max(0, (int)std::ceil(center(0) - r));
      int j_lower = std::max(0, (int)std::ceil(center(1) - r));
      int i_higher = std::min((int)std::floor(center(0) + r), canvas_width - 1);
      int j_higher =
          std::min((int)std::floor(center(1) + r), canvas_height - 1);
      const auto w = _color.w;
      for (int i = i_lower; i <= i_higher; i++) {
        for (int j = j_lower; j <= j_higher; j++) {
          real dist = length(center - Vector2(i, j));
          auto alpha = w * clamp(r - dist);
          auto &dest = canvas.img[Vector2i(i, j)];
          dest = lerp(alpha, dest, _color);
        }
      }
    }

    TI_FORCE_INLINE ~Circle() {
      if (!finished)
        finish();
    }
  };

 public:
  Array2D<Vector4> &img;
  Matrix3 transform_matrix;

  Canvas(Array2D<Vector4> &img) : img(img) {
    transform_matrix = Matrix3(Vector3(img.get_res().cast<real>(), 1.0_f));
  }

  TI_FORCE_INLINE Vector2 transform(Vector2 x) const {
    return Vector2(transform_matrix * Vector3(x, 1.0_f));
  }

  TI_FORCE_INLINE Vector2 untransform(Vector2 x) const {
    return Vector2(inversed(transform_matrix) * Vector3(x, 1.0_f));
  }

  std::vector<Circle> circles;
  std::vector<Line> lines;

  Circle &circle(Vector2 center) {
    circles.emplace_back(*this, center);
    return circles.back();
  }

  Circle &circle(real x, real y) {
    circles.emplace_back(*this, Vector2(x, y));
    return circles.back();
  }

  void circles_batched(int n,
                       std::size_t x,
                       uint32 color_single,
                       std::size_t color_array,
                       real radius_single,
                       std::size_t radius_array);

  void circle_single(real x, real y, uint32 color, real radius);

  void paths_batched(int n,
                     std::size_t a_,
                     std::size_t b_,
                     uint32 color_single,
                     std::size_t color_array,
                     real radius_single,
                     std::size_t radius_array);

  void path_single(real x0,
                   real y0,
                   real x1,
                   real y1,
                   uint32 color,
                   real radius);

  Line &path(real xa, real ya, real xb, real yb) {
    return path(Vector2(xa, ya), Vector2(xb, yb));
  }

  Line &path() {
    lines.emplace_back(*this);
    return lines.back();
  }

  Line &path(Vector2 a, Vector2 b) {
    lines.emplace_back(*this);
    lines.back().path(a, b);
    return lines.back();
  }

  Line &path(Vector2 a, Vector2 b, Vector2 c) {
    lines.emplace_back(*this, a, b, c);
    return lines.back();
  }

  Line &path(Vector2 a, Vector2 b, Vector2 c, Vector2 d) {
    lines.emplace_back(*this, a, b, c, d);
    return lines.back();
  }

  Line &rect(Vector2 a, Vector2 b) {
    lines.emplace_back(*this, a, Vector2(a.x, b.y), b, Vector2(b.x, a.y));
    return lines.back();
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

  void triangle(Vector2 a, Vector2 b, Vector2 c, Vector4 color);

  void triangles_batched(int n,
                         std::size_t a_,
                         std::size_t b_,
                         std::size_t c_,
                         uint32 color_single,
                         std::size_t color_array);

  void triangle_single(real x0,
                       real y0,
                       real x1,
                       real y1,
                       real x2,
                       real y2,
                       uint32 color);

  void text(const std::string &str,
            Vector2 position,
            real size,
            Vector4 color) {
    position = transform(position);
    std::string folder;
    folder = fmt::format("{}/../../assets", lang::runtime_lib_dir());
    auto ttf_path = fmt::format("{}/Go-Regular.ttf", folder);
    img.write_text(ttf_path, str, size, position.x, position.y, color);
  }

  void clear(Vector4 color) {
    circles.clear();
    lines.clear();
    img.reset(color);
  }

  void clear(uint32 c) {
    clear(color_from_hex(c));
  }

  ~Canvas() {
  }

  void set_identity_transform_matrix() {
    transform_matrix = Matrix3(1);
  }
};

#if defined(TI_GUI_ANDROID)

class GUIBaseAndroid {
 public:
  // @TODO
};

using GUIBase = GUIBaseAndroid;

#endif

#if defined(TI_GUI_X11)

class CXImage;

class GUIBaseX11 {
 public:
  void *display;
  void *visual;
  unsigned long window;
  CXImage *img;
  std::vector<char> wmDeleteMessage;
};

using GUIBase = GUIBaseX11;

#endif

#if defined(TI_GUI_WIN32)
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

#if defined(TI_GUI_COCOA)
class GUIBaseCocoa {
 public:
  id window, view;
  std::size_t img_data_length;
  std::vector<uint8_t> img_data;
  std::atomic_bool window_received_close = false;
  // Some key are called *modifier keys* and are not detected by regular key
  // events in Cocoa. Instead, they will trigger a flags changed event.
  // https://developer.apple.com/documentation/appkit/nseventtype/nseventtypeflagschanged?language=objc
  //
  // We have to:
  // 1. check [event modifierFlags] to retrieve the key code,
  // 2. maintain the press/released events on our own.
  std::unordered_map<std::string, bool> active_modifier_flags;
};

using GUIBase = GUIBaseCocoa;
#endif

class TI_DLL_EXPORT GUI : public GUIBase {
 public:
  std::string window_name;
  int width, height;
  int frame_id = 0;
  real frame_delta_limit = 1.0 / 60;
  float64 start_time;
  Array2D<Vector4> buffer;
  std::vector<real> last_frame_interval;
  std::unique_ptr<Canvas> canvas;
  float64 last_frame_time;
  bool key_pressed;
  int should_close{0};
  std::vector<std::string> log_entries;
  Vector2i cursor_pos;
  bool button_status[3];
  int widget_height;
  std::vector<std::unique_ptr<float>> widget_values;
  bool show_gui;
  bool fullscreen;
  bool fast_gui;
  uintptr_t fast_buf;

  void set_mouse_pos(int x, int y) {
    cursor_pos = Vector2i(x, y);
  }

  Vector2i widget_size = Vector2i(200, 40);

  struct MouseEvent {
    enum class Type { move, press, release };
    Type type;
    Vector2i pos;
    bool button_status[3];
  };

  struct KeyEvent {
    enum class Type { move, press, release };
    Type type;
    std::string key;
    Vector2i pos;
    Vector2i delta;
  };

  std::vector<KeyEvent> key_events;

  struct Rect {
    Vector2i pos;
    Vector2i size;
    TI_IO_DEF(pos, size);
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
    bool hover;

    Widget() {
      hover = false;
    }

    Widget(Rect rect) : Widget() {
      this->rect = rect;
    }

    bool inside(Vector2i p) {
      return rect.inside(p);
    }

    virtual void mouse_event(MouseEvent e) {
    }

    virtual void redraw(Canvas &canvas) {
      Vector4 color =
          hover ? color_from_hex(widget_bg) : color_from_hex(widget_hover);
      for (int i = 1; i < rect.size[0] - 1; i++) {
        for (int j = 1; j < rect.size[1] - 1; j++) {
          canvas.img[rect.pos[0] + i][rect.pos[1] + j] = color;
        }
      }
    }

    void set_hover(bool val) {
      hover = val;
    }

    virtual ~Widget() {
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
          color_from_hex(text_color));
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

    const int slider_padding = 10;

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
      if ((e.type == MouseEvent::Type::press ||
           e.type == MouseEvent::Type::move) &&
          e.button_status[0]) {
        real alpha = clamp(real(e.pos[0] - rect.pos[0] - slider_padding) /
                           (rect.size[0] - slider_padding * 2));
        real offset = 0.0_f;
        if (std::is_integral<T>::value) {
          offset = 0.5_f;
        }
        val = static_cast<T>(alpha * (maximum - minimum) + minimum + offset);
      }
    }

    void redraw(Canvas &canvas) override {
      Widget::redraw(canvas);
      int s = 16;
      auto text_with_value = text;
      if (std::is_integral<T>::value) {
        text_with_value += fmt::format(": {}", val);
      } else {
        text_with_value += fmt::format(": {:.3f}", val);
      }
      canvas.text(
          text_with_value,
          (rect.pos + Vector2i(2, rect.size[1] - 2)).template cast<real>(), s,
          color_from_hex(text_color));
      int slider_start = slider_padding,
          slider_end = rect.size[0] - slider_padding;
      for (int i = slider_start; i < slider_end; i++) {
        for (int j = slider_padding; j < slider_padding + 3; j++) {
          canvas.img[rect.pos[0] + i][rect.pos[1] + j] =
              color_from_hex(slider_bar_color);
        }
      }
      auto alpha = (val - minimum) / real(maximum - minimum);
      canvas
          .circle(rect.pos.template cast<real>() +
                  Vector2(lerp(alpha, slider_start, slider_end),
                          slider_padding + 1))
          .radius(5)
          .color(color_from_hex(slider_circle_color));
    }
  };

  template <typename T>
  class Label : public Widget {
   public:
    std::string text;
    T &val;

    const int slider_padding = 5;

    Label(Rect rect, const std::string text, T &val)
        : Widget(rect), text(text), val(val) {
    }

    void redraw(Canvas &canvas) override {
      Widget::redraw(canvas);
      int s = 16;
      auto text_with_value = text;
      if (std::is_integral<T>::value) {
        text_with_value += fmt::format(": {}", val);
      } else {
        text_with_value += fmt::format(": {:.3f}", val);
      }
      canvas.text(
          text_with_value,
          (rect.pos + Vector2i(2, rect.size[1] - 2)).template cast<real>(), s,
          color_from_hex(text_color));
    }
  };

  Rect make_widget_rect(int h) {
    widget_height += h;
    return Rect(Vector2i(width - widget_size[0], height - widget_height),
                Vector2i(widget_size[0], h));
  }

  GUI &button(std::string text, const Button::CallbackType &callback) {
    widgets.push_back(std::make_unique<Button>(make_widget_rect(widget_size[1]),
                                               text, callback));
    return *this;
  }

  template <typename T>
  GUI &slider(std::string text, T &val, T minimum, T maximum, T step = 1) {
    widgets.push_back(std::make_unique<Slider<T>>(
        make_widget_rect(widget_size[1]), text, val, minimum, maximum, step));
    return *this;
  }

  template <typename T>
  GUI &label(std::string text, T &val) {
    widgets.push_back(std::make_unique<Label<T>>(
        make_widget_rect(widget_size[1] / 2), text, val));
    return *this;
  }

  void process_event();

  void send_window_close_message() {
    key_events.push_back(
        GUI::KeyEvent{GUI::KeyEvent::Type::press, "WMClose", cursor_pos});
    should_close++;
  }

  void mouse_event(MouseEvent e) {
    if (e.type == MouseEvent::Type::press) {
      button_status[0] = true;
    }
    if (e.type == MouseEvent::Type::release) {
      button_status[0] = false;
    }
    e.button_status[0] = button_status[0];
    for (auto &w : widgets) {
      if (w->inside(e.pos)) {
        w->mouse_event(e);
      }
    }
  }

  explicit GUI(const std::string &window_name,
               int width = 800,
               int height = 800,
               bool show_gui = true,
               bool fullscreen = true,
               bool fast_gui = false,
               uintptr_t fast_buf = 0,
               bool normalized_coord = true)
      : window_name(window_name),
        width(width),
        height(height),
        key_pressed(false),
        show_gui(show_gui),
        fullscreen(fullscreen),
        fast_gui(fast_gui),
        fast_buf(fast_buf) {
    memset(button_status, 0, sizeof(button_status));
    start_time = taichi::Time::get_time();
    buffer.initialize(Vector2i(width, height));
    canvas = std::make_unique<Canvas>(buffer);
    last_frame_time = taichi::Time::get_time();
    if (!normalized_coord) {
      canvas->set_identity_transform_matrix();
    }
    widget_height = 0;
    if (show_gui) {
      initialize_window();
    }
  }

  explicit GUI(const std::string &window_name,
               Vector2i res,
               bool show_gui,
               bool fullscreen = true,
               bool fast_gui = false,
               uintptr_t fast_buf = 0,
               bool normalized_coord = true)
      : GUI(window_name,
            res[0],
            res[1],
            show_gui,
            fullscreen,
            fast_gui,
            fast_buf,
            normalized_coord) {
  }

  void create_window();

  void initialize_window() {
    create_window();
    set_title(window_name);
  }

  Canvas &get_canvas() {
    return *canvas;
  }

  void redraw();

  void set_title(std::string title);

  void redraw_widgets() {
    auto old_transform_matrix = canvas->transform_matrix;
    canvas->set_identity_transform_matrix();
    for (auto &w : widgets) {
      w->set_hover(w->inside(cursor_pos));
      w->redraw(*canvas);
    }
    canvas->transform_matrix = old_transform_matrix;
  }

  void update() {
    frame_id++;
    if (show_gui) {
      taichi::Time::wait_until(last_frame_time + frame_delta_limit);
      auto this_frame_time = taichi::Time::get_time();
      if (last_frame_time != 0) {
        last_frame_interval.push_back(this_frame_time - last_frame_time);
      }
      last_frame_time = this_frame_time;
      // Some old examples / users don't even provide a `break` statement for us
      // to terminate loop. So we have to terminate the program with
      // RuntimeError if ti.GUI.EXIT event is not processed. Pretty like
      // SIGTERM, you can hook it, but you have to terminate after your handler
      // is done.
      if (should_close) {
        if (++should_close > 5) {
          // if the event is not processed in 5 frames, raise RuntimeError
          throw std::string(
              "Window close button clicked, exiting... (use `while "
              "gui.running` "
              "to exit gracefully)");
        }
      }
      while (last_frame_interval.size() > 30) {
        last_frame_interval.erase(last_frame_interval.begin());
      }
      auto real_fps = last_frame_interval.size() /
                      (std::accumulate(last_frame_interval.begin(),
                                       last_frame_interval.end(), 0.0_f));
      redraw_widgets();
      redraw();
      process_event();
      if (frame_id % 10 == 0)
        set_title(fmt::format("{} ({:.2f} FPS)", window_name, real_fps));
    }
  }

  bool has_key_event() {
    return !!key_events.size();
  }

  void wait_key_event() {
    while (!key_events.size()) {
      update();
    }
  }

  KeyEvent get_key_event_head() {
    return key_events[0];
  }

  Vector2 get_cursor_pos() {
    return canvas->untransform(Vector2(cursor_pos));
  }

  void pop_key_event_head() {
    key_events.erase(key_events.begin());
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

  Vector2 canvas_untransform(Vector2i pos) {
    return canvas->untransform(Vector2(pos));
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

  void screenshot(std::string filename = "") {
    if (filename == "") {
      char timestamp[80];
      std::time_t t = std::time(nullptr);
      std::tm tstruct = *localtime(&t);
      std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S",
                    &tstruct);
      filename = std::string(timestamp) + ".png";
    }
    canvas->img.write_as_image(filename);
  }

  ~GUI();
};

TI_NAMESPACE_END
