#include <taichi/visual/gui.h>
#include <numeric>

#if defined(TC_PLATFORM_LINUX)
#include <X11/Xlib.h>
#include <X11/Xutil.h>

// Undo terrible unprefixed macros in X.h
#ifdef None
#undef None
#endif
#ifdef Success
#undef Success
#endif

TC_NAMESPACE_BEGIN

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

  void set_data(const Array2D<Vector4> &color) {
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

void GUI::process_event() {
  while (XPending((Display *)display)) {
    XEvent ev;
    XNextEvent((Display *)display, &ev);
    switch (ev.type) {
      case Expose:
        break;
      case ButtonPress:
        // exit(0);
        break;
    }
  }
}

GUI::GUI(const std::string &window_name, int width, int height)
    : window_name(window_name), width(width), height(height) {
  display = XOpenDisplay(NULL);
  visual = DefaultVisual(display, 0);
  window =
      XCreateSimpleWindow((Display *)display, RootWindow((Display *)display, 0),
                          0, 0, width, height, 1, 0, 0);
  XStoreName((Display *)display, window, window_name.c_str());
  XSelectInput((Display *)display, window, ButtonPressMask | ExposureMask);
  XMapWindow((Display *)display, window);
  img = std::make_unique<CXImage>((Display *)display, (Visual *)visual, width,
                                  height);
  start_time = taichi::Time::get_time();
  buffer.initialize(Vector2i(width, height));
  canvas = std::make_unique<Canvas>(buffer);
  last_frame_time = taichi::Time::get_time();
}

void GUI::update() {
  img->set_data(buffer);
  frame_id++;
  while (taichi::Time::get_time() < start_time + frame_id / (real)fps) {
  }
  XPutImage((Display *)display, window, DefaultGC(display, 0), img->image, 0, 0,
            0, 0, width, height);
  process_event();
  while (last_frame_interval.size() > 30) {
    last_frame_interval.erase(last_frame_interval.begin());
  }
  auto real_fps = last_frame_interval.size() /
                  (std::accumulate(last_frame_interval.begin(),
                                   last_frame_interval.end(), 0.0_f));
  XStoreName((Display *)display, window,
             fmt::format("{} ({:.04f} FPS)", window_name, real_fps).c_str());
  if (last_frame_time != 0) {
    last_frame_interval.push_back(taichi::Time::get_time() - last_frame_time);
  }
  last_frame_time = taichi::Time::get_time();
}

GUI::~GUI() {
}

TC_NAMESPACE_END
#endif