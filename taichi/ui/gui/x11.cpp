#include "taichi/ui/gui/gui.h"

#if defined(TI_GUI_X11)
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <cstdlib>

// Undo terrible unprefixed macros in X.h
#ifdef None
#undef None
#endif
#ifdef Success
#undef Success
#endif

TI_NAMESPACE_BEGIN

class CXImage {
 public:
  XImage *image;
  std::vector<uint8> image_data;
  void *fast_data{nullptr};
  int width, height;

  CXImage(Display *display, Visual *visual, int width, int height)
      : width(width), height(height) {
    image_data.resize(width * height * 4);
    image = XCreateImage(display, visual, 24, ZPixmap, 0,
                         (char *)image_data.data(), width, height, 32, 0);
    TI_ASSERT((void *)image->data == image_data.data());
  }

  CXImage(Display *display,
          Visual *visual,
          void *fast_data,
          int width,
          int height)
      : width(width), height(height) {
    image = XCreateImage(display, visual, 24, ZPixmap, 0, (char *)fast_data,
                         width, height, 32, 0);
    TI_ASSERT((void *)image->data == fast_data);
  }

  void set_data(const Array2D<Vector4> &color) {
    auto p = image_data.data();
    for (int j = 0; j < height; j++) {
      for (int i = 0; i < width; i++) {
        auto c = color[i][height - j - 1];
        *p++ = uint8(clamp(int(c[2] * 255.0_f), 0, 255));
        *p++ = uint8(clamp(int(c[1] * 255.0_f), 0, 255));
        *p++ = uint8(clamp(int(c[0] * 255.0_f), 0, 255));
        *p++ = 0;
      }
    }
  }

  ~CXImage() {
    delete image;  // image->data is automatically released in image_data
  }
};

static std::string lookup_keysym(XEvent *ev) {
  int key = XLookupKeysym(&ev->xkey, 0);
  if (isascii(key))
    return std::string(1, key);
  else
    return XKeysymToString(key);
}

static std::string lookup_button(XEvent *ev) {
  switch (ev->xbutton.button) {
    case 1:
      return "LMB";
    case 2:
      return "MMB";
    case 3:
      return "RMB";
    default:
      return fmt::format("Button{}", ev->xbutton.button);
  }
}

void GUI::process_event() {
  while (XPending((Display *)display)) {
    XEvent ev;
    XNextEvent((Display *)display, &ev);
    switch (ev.type) {
      case Expose:
        break;
      case ClientMessage:
        // https://stackoverflow.com/questions/10792361/how-do-i-gracefully-exit-an-x11-event-loop
        if (ev.xclient.data.l[0] == *(Atom *)wmDeleteMessage.data()) {
          send_window_close_message();
        }
        break;
      case MotionNotify:
        set_mouse_pos(ev.xbutton.x, height - ev.xbutton.y - 1);
        mouse_event(MouseEvent{MouseEvent::Type::move, cursor_pos});
        key_events.push_back(
            KeyEvent{KeyEvent::Type::move, "Motion", cursor_pos});
        break;
      case ButtonPress:
        set_mouse_pos(ev.xbutton.x, height - ev.xbutton.y - 1);
        mouse_event(MouseEvent{MouseEvent::Type::press, cursor_pos});
        switch (ev.xbutton.button) {
          case 4:  // wheel up
            key_events.push_back(KeyEvent{KeyEvent::Type::move, "Wheel",
                                          cursor_pos, Vector2i{0, +120}});
            break;
          case 5:  // wheel down
            key_events.push_back(KeyEvent{KeyEvent::Type::move, "Wheel",
                                          cursor_pos, Vector2i{0, -120}});
            break;
          case 6:  // wheel right
            key_events.push_back(KeyEvent{KeyEvent::Type::move, "Wheel",
                                          cursor_pos, Vector2i{+120, 0}});
            break;
          case 7:  // wheel left
            key_events.push_back(KeyEvent{KeyEvent::Type::move, "Wheel",
                                          cursor_pos, Vector2i{-120, 0}});
            break;
          default:  // normal mouse button
            key_events.push_back(KeyEvent{KeyEvent::Type::press,
                                          lookup_button(&ev), cursor_pos});
            break;
        }
        break;
      case ButtonRelease:
        set_mouse_pos(ev.xbutton.x, height - ev.xbutton.y - 1);
        mouse_event(MouseEvent{MouseEvent::Type::release, cursor_pos});
        key_events.push_back(
            KeyEvent{KeyEvent::Type::release, lookup_button(&ev), cursor_pos});
        break;
      case KeyPress:
        key_pressed = true;
        key_events.push_back(
            KeyEvent{KeyEvent::Type::press, lookup_keysym(&ev), cursor_pos});
        break;
      case KeyRelease:
        key_events.push_back(
            KeyEvent{KeyEvent::Type::release, lookup_keysym(&ev), cursor_pos});
        break;
    }
  }
}

void GUI::create_window() {
  display = XOpenDisplay(nullptr);
  TI_ASSERT_INFO(display,
                 "Taichi fails to create a window."
                 " This is probably due to the lack of an X11 GUI environment."
                 " Consider using the `ti.GUI(show_gui=False)` option, see"
                 " https://docs.taichi-lang.org/docs/gui_system");
  visual = DefaultVisual(display, 0);
  window =
      XCreateSimpleWindow((Display *)display, RootWindow((Display *)display, 0),
                          0, 0, width, height, 1, 0, 0);
  TI_ASSERT_INFO(window, "failed to create X window");

  if (fullscreen) {
    // https://stackoverflow.com/questions/9083273/x11-fullscreen-window-opengl
    Atom atoms[2] = {
        XInternAtom((Display *)display, "_NET_WM_STATE_FULLSCREEN", False), 0};
    Atom wmstate = XInternAtom((Display *)display, "_NET_WM_STATE", False);
    XChangeProperty((Display *)display, window, wmstate, XA_ATOM, 32,
                    PropModeReplace, (unsigned char *)atoms, 1);
  }

  XSelectInput((Display *)display, window,
               ButtonPressMask | ExposureMask | KeyPressMask | KeyReleaseMask |
                   ButtonPress | ButtonReleaseMask | EnterWindowMask |
                   LeaveWindowMask | PointerMotionMask);
  wmDeleteMessage = std::vector<char>(sizeof(Atom));
  *(Atom *)wmDeleteMessage.data() =
      XInternAtom((Display *)display, "WM_DELETE_WINDOW", False);
  XSetWMProtocols((Display *)display, window, (Atom *)wmDeleteMessage.data(),
                  1);
  XMapWindow((Display *)display, window);
  if (!fast_gui)
    img = new CXImage((Display *)display, (Visual *)visual, width, height);
  else
    img = new CXImage((Display *)display, (Visual *)visual, (void *)fast_buf,
                      width, height);
}

void GUI::redraw() {
  if (!fast_gui)
    img->set_data(buffer);
  XPutImage((Display *)display, window, DefaultGC(display, 0), img->image, 0, 0,
            0, 0, width, height);
}

void GUI::set_title(std::string title) {
  XStoreName((Display *)display, window, title.c_str());
}

GUI::~GUI() {
  if (show_gui) {
    XCloseDisplay((Display *)display);
    delete img;
  }
}

TI_NAMESPACE_END

#endif
