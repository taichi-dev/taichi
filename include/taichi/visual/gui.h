#include <taichi/taichi>
#include <numeric>

#if defined(TC_PLATFORM_LINUX)
#include <X11/Xlib.h>
#include <X11/Xutil.h>

TC_NAMESPACE_BEGIN

class Canvas {
 public:
  Array2D<Vector4> &img;
  Matrix3 transform_matrix;

  Canvas(Array2D<Vector4> &img) : img(img) {
    transform_matrix = Matrix3(Vector3(img.get_res().cast<real>(), 1.0_f));
  }

  TC_FORCE_INLINE Vector2 transform(Vector2 x) {
    return Vector2(transform_matrix * Vector3(x, 1.0_f));
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
    for (int i = std::floor(limits[0].x); i < std::ceil(limits[1].x); i++) {
      for (int j = std::floor(limits[0].y); j < std::ceil(limits[1].y); j++) {
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

  void clear(Vector4 color) {
    img.reset(color);
  }
};

class GUI {
 public:
  class CXImage {
   public:
    XImage *image;
    std::vector<uint8> image_data;
    int width, height;
    CXImage(Display *display, Visual *visual, int width, int height)
        : width(width), height(height) {
      image_data.resize(width * height * 4);
      image = XCreateImage(display, visual, 24, ZPixmap, 0,
                           (char *)image_data.data(), width, height, 32, 0);
      TC_ASSERT((void *)image->data == image_data.data());
    }

    void set_frame(const Array2D<Vector4> &color) {
      auto p = image_data.data();
      for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
          auto c = color[i][height - j - 1];
          *p++ = uint8(clamp(int(c[0] * 255.0_f), 0, 255));
          *p++ = uint8(clamp(int(c[1] * 255.0_f), 0, 255));
          *p++ = uint8(clamp(int(c[2] * 255.0_f), 0, 255));
          *p++ = uint8(clamp(int(c[3] * 255.0_f), 0, 255));
        }
      }
    }

    ~CXImage() {
      delete image;  // image->data is automatically released in image_data
    }
  };

  std::string window_name;
  int width, height;
  Display *display;
  Visual *visual;
  Window window;
  std::unique_ptr<CXImage> img;
  int frame_id = 0;
  const int fps = 60;
  float64 start_time;
  Array2D<Vector4> buffer;
  std::vector<real> last_frame_interval;
  std::unique_ptr<Canvas> canvas;
  float64 last_frame_time;

  void process_event() {
    while (XPending(display)) {
      XEvent ev;
      XNextEvent(display, &ev);
      switch (ev.type) {
        case Expose:
          break;
        case ButtonPress:
          // exit(0);
          break;
      }
    }
  }

  GUI(const std::string &window_name, int width, int height)
      : window_name(window_name), width(width), height(height) {
    display = XOpenDisplay(NULL);
    visual = DefaultVisual(display, 0);
    window = XCreateSimpleWindow(display, RootWindow(display, 0), 0, 0, width,
                                 height, 1, 0, 0);
    XStoreName(display, window, window_name.c_str());
    XSelectInput(display, window, ButtonPressMask | ExposureMask);
    XMapWindow(display, window);
    img = std::make_unique<CXImage>(display, visual, width, height);
    start_time = taichi::Time::get_time();
    buffer.initialize(Vector2i(width, height));
    canvas = std::make_unique<Canvas>(buffer);
    last_frame_time = taichi::Time::get_time();
  }

  Canvas &get_canvas() {
    return *canvas;
  }

  void clear_buffer(Vector4 color = Vector4(1)) {
    buffer.reset(color);
  }

  void update() {
    img->set_frame(buffer);
    frame_id++;
    while (taichi::Time::get_time() < start_time + frame_id / (real)fps) {
    }
    XPutImage(display, window, DefaultGC(display, 0), img->image, 0, 0, 0, 0,
              width, height);
    process_event();
    while (last_frame_interval.size() > 30) {
      last_frame_interval.erase(last_frame_interval.begin());
    }
    auto real_fps = last_frame_interval.size() /
                    (std::accumulate(last_frame_interval.begin(),
                                     last_frame_interval.end(), 0.0_f));
    XStoreName(display, window,
               fmt::format("{} ({:.04f} FPS)", window_name, real_fps).c_str());
    if (last_frame_time != 0) {
      last_frame_interval.push_back(taichi::Time::get_time() - last_frame_time);
    }
    last_frame_time = taichi::Time::get_time();
  }

  ~GUI() {
  }
};

TC_NAMESPACE_END
#endif
