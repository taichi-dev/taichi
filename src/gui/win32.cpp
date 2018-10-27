#include <taichi/common/util.h>

#if defined(TC_PLATFORM_WINDOWS)

#include <taichi/common/task.h>
#include <taichi/visual/gui.h>
#include <map>

// Note: some code is copied from MSDN:
// https://docs.microsoft.com/en-us/windows/desktop/learnwin32/introduction-to-windows-programming-in-c--

std::map<HWND, taichi::GUI *> gui_from_hwnd;

LRESULT CALLBACK WindowProc(HWND hwnd,
                            UINT uMsg,
                            WPARAM wParam,
                            LPARAM lParam) {
  auto dc = GetDC(hwnd);

  switch (uMsg) {
    case WM_DESTROY:
      PostQuitMessage(0);
      exit(0);
      return 0;
    case WM_PAINT: {
    }
     return 0;
  }
  return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

TC_NAMESPACE_BEGIN

void GUI::process_event() {
  MSG msg;
  if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
}

GUI::GUI(const std::string &window_name, int width, int height, bool normalized_coord)
    : window_name(window_name),
      width(width),
      height(height),
      key_pressed(false) {
  const char CLASS_NAME[] = "Taichi Win32 Window";

  WNDCLASS wc = {};

  wc.lpfnWndProc = WindowProc;
  wc.hInstance = GetModuleHandle(0);
  wc.lpszClassName = CLASS_NAME;

  RegisterClass(&wc);

  hwnd = CreateWindowEx(0,                    // Optional window styles.
                        CLASS_NAME,           // Window class
                        window_name.c_str(),  // Window text
                        WS_OVERLAPPEDWINDOW,  // Window style
                        // Size and position
                        CW_USEDEFAULT, CW_USEDEFAULT, width + 16, height + 32,
                        NULL,                // Parent window
                        NULL,                // Menu
                        GetModuleHandle(0),  // Instance handle
                        NULL                 // Additional application data
  );

  gui_from_hwnd[hwnd] = this;

  if (hwnd == NULL) {
    TC_ERROR("Window creation failed");
  }

  ShowWindow(hwnd, SW_SHOWDEFAULT);

  hdc = GetDC(hwnd);

  start_time = taichi::Time::get_time();
  buffer.initialize(Vector2i(width, height));
  canvas = std::make_unique<Canvas>(buffer);
  last_frame_time = taichi::Time::get_time();
  data = (COLORREF *)calloc(width * height, sizeof(COLORREF));
  src = CreateCompatibleDC(hdc);
  if (!normalized_coord) {
    canvas->set_idendity_transform_matrix();
  }
}

void GUI::redraw() {
  // http:// www.cplusplus.com/reference/cstdlib/calloc/
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      auto c = reinterpret_cast<unsigned char *>(data + (j * width) + i);
      c[0] = (unsigned char)(canvas->img[i][height - j - 1][2] * 255.0_f);
      c[1] = (unsigned char)(canvas->img[i][height - j - 1][1] * 255.0_f);
      c[2] = (unsigned char)(canvas->img[i][height - j - 1][0] * 255.0_f);
      c[3] = 0;
    }
  }
  bitmap = CreateBitmap(width, height, 1, 32, (void *)data);
  SelectObject(src, bitmap);
  BitBlt(hdc, 0, 0, width, height, src, 0, 0, SRCCOPY);
  DeleteObject(bitmap);
}

void GUI::update() {
  frame_id++;
  while (taichi::Time::get_time() < start_time + frame_id / (real)fps) {
  }
  redraw();
  UpdateWindow(hwnd);
  process_event();
  while (last_frame_interval.size() > 30) {
    last_frame_interval.erase(last_frame_interval.begin());
  }
  auto real_fps = last_frame_interval.size() /
                  (std::accumulate(last_frame_interval.begin(),
                                   last_frame_interval.end(), 0.0_f));

  SetWindowText(hwnd, fmt::format("{} ({:.02f} FPS)", window_name, real_fps).c_str());
  if (last_frame_time != 0) {
    last_frame_interval.push_back(taichi::Time::get_time() - last_frame_time);
  }
  last_frame_time = taichi::Time::get_time();
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
  std::free(data);
  DeleteDC(src);
  gui_from_hwnd.erase(hwnd);
}

#if !defined(TC_AMALGAMATED)
auto win32guitest = []() {
  GUI gui("Test2", 800, 300);
  auto &canvas = gui.get_canvas();
  canvas.clear(Vector4(0, 1, 0, 0));
  for (int i = 0; i < 2000; i++) {
    canvas.line(Vector2(0, 0), Vector2(1, i * 0.01_f),
                          Vector4(0, 0, i * 0.01_f, 0));
    gui.update();
  }
};
TC_REGISTER_TASK(win32guitest);
#endif

TC_NAMESPACE_END

#endif