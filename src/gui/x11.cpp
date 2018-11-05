#include <taichi/visual/gui.h>

#if defined(TC_GUI_X11)
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
        *p++ = uint8(clamp(int(c[2] * 255.0_f), 0, 255));
        *p++ = uint8(clamp(int(c[1] * 255.0_f), 0, 255));
        *p++ = uint8(clamp(int(c[0] * 255.0_f), 0, 255));
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
      case MotionNotify:
        set_mouse_pos(ev.xbutton.x, height - ev.xbutton.y - 1);
        mouse_event(MouseEvent{MouseEvent::Type::move, cursor_pos});
        break;
      case ButtonPress:
        mouse_event(MouseEvent{MouseEvent::Type::press, cursor_pos});
        break;
      case ButtonRelease:
        mouse_event(MouseEvent{MouseEvent::Type::release, cursor_pos});
        break;
      case KeyPress:
        key_pressed = true;
        break;
    }
  }
}

void GUI::create_window() {
  display = XOpenDisplay(nullptr);
  visual = DefaultVisual(display, 0);
  window =
      XCreateSimpleWindow((Display *)display, RootWindow((Display *)display, 0),
                          0, 0, width, height, 1, 0, 0);
  XSelectInput((Display *)display, window,
               ButtonPressMask | ExposureMask | KeyPressMask | KeyReleaseMask |
                   ButtonPress | ButtonReleaseMask | EnterWindowMask |
                   LeaveWindowMask | PointerMotionMask);
  XMapWindow((Display *)display, window);
  img = new CXImage((Display *)display, (Visual *)visual, width, height);
}

void GUI::redraw() {
  img->set_data(buffer);
  XPutImage((Display *)display, window, DefaultGC(display, 0), img->image, 0, 0,
            0, 0, width, height);
}

void GUI::set_title(std::string title) {
  XStoreName((Display *)display, window, title.c_str());
}

GUI::~GUI() {
  delete img;
}

TC_NAMESPACE_END
#endif