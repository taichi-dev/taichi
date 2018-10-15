#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/visual/gui.h>

// Note: some code is copied from MSDN:
// https://docs.microsoft.com/en-us/windows/desktop/learnwin32/introduction-to-windows-programming-in-c--

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

int width = 1280;
int height = 800;

void update(HDC hdc) {
  // http:// www.cplusplus.com/reference/cstdlib/calloc/
  COLORREF *data = (COLORREF *)calloc(width * height, sizeof(COLORREF));
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      auto c = reinterpret_cast<unsigned char *>(data + (j * width) + i);
      c[0] = j % 255;
      c[1] = i % 255;
      c[2] = 0;
      c[3] = 0;
    }
  }

  HBITMAP bitmap = CreateBitmap(width, height, 1, 32, (void *)data);
  HDC src = CreateCompatibleDC(hdc);
  SelectObject(src, bitmap);
  BitBlt(hdc, 0, 0, width, height, src, 0, 0, SRCCOPY);
  std::free(data);
  DeleteDC(src);
}

int create_window(std::string window_name) {
  // Register the window class.
  const char CLASS_NAME[] = "Taichi Win32 Window";

  WNDCLASS wc = {};

  wc.lpfnWndProc = WindowProc;
  wc.hInstance = GetModuleHandle(0);
  wc.lpszClassName = CLASS_NAME;

  RegisterClass(&wc);

  HWND hwnd =
      CreateWindowEx(0,                           // Optional window styles.
                     CLASS_NAME,                  // Window class
                     window_name.c_str(),         // Window text
                     WS_OVERLAPPEDWINDOW,         // Window style
                     // Size and position
                     CW_USEDEFAULT, CW_USEDEFAULT, width + 16, height + 32,

                     NULL,                // Parent window
                     NULL,                // Menu
                     GetModuleHandle(0),  // Instance handle
                     NULL                 // Additional application data
      );

  if (hwnd == NULL) {
    return 0;
  }

  ShowWindow(hwnd, SW_SHOWDEFAULT);

  MSG msg = {};
  while (GetMessage(&msg, NULL, 0, 0)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }

  return 0;
}

LRESULT CALLBACK WindowProc(HWND hwnd,
                            UINT uMsg,
                            WPARAM wParam,
                            LPARAM lParam) {
  auto dc = GetDC(hwnd);
  switch (uMsg) {
    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;

    case WM_PAINT: {
      update(dc);
    }
    return 0;
  }
  return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

TC_NAMESPACE_BEGIN

void GUI::process_event() {
  /*
  while (XPending((Display *)display)) {
    XEvent ev;
    XNextEvent((Display *)display, &ev);
    switch (ev.type) {
      case Expose:
        break;
      case ButtonPress:
        break;
      case KeyPress:
        key_pressed = true;
        break;
    }
  }
  */
}

GUI::GUI(const std::string &window_name, int width, int height)
    : window_name(window_name),
      width(width),
      height(height),
      key_pressed(false) {
  /*
  display = XOpenDisplay(nullptr);
  visual = DefaultVisual(display, 0);
  window =
      XCreateSimpleWindow((Display *)display, RootWindow((Display *)display, 0),
                          0, 0, width, height, 1, 0, 0);
  XStoreName((Display *)display, window, window_name.c_str());
  XSelectInput((Display *)display, window,
               ButtonPressMask | ExposureMask | KeyPressMask | KeyReleaseMask);
  XMapWindow((Display *)display, window);
  img = std::make_unique<CXImage>((Display *)display, (Visual *)visual, width,
                                  height);
  start_time = taichi::Time::get_time();
  buffer.initialize(Vector2i(width, height));
  canvas = std::make_unique<Canvas>(buffer);
  last_frame_time = taichi::Time::get_time();
*/
}

void GUI::update() {
  /*
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
             fmt::format("{} ({:.02f} FPS)", window_name, real_fps).c_str());
  if (last_frame_time != 0) {
    last_frame_interval.push_back(taichi::Time::get_time() - last_frame_time);
  }
  last_frame_time = taichi::Time::get_time();
  */
}

void GUI::wait_key() {
  while (true) {
    key_pressed = false;
    update();
    if (key_pressed) {
      break;
    }
  }
}

GUI::~GUI() {
}


auto win32guitest = []() { create_window("Taichi Win32 Window Test"); };

TC_REGISTER_TASK(win32guitest);

TC_NAMESPACE_END