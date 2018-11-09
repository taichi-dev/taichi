#include <taichi/common/util.h>

#if defined(TC_PLATFORM_WINDOWS)
#include <windowsx.h>
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
  auto gui = gui_from_hwnd[hwnd];
  using namespace taichi;
  int x, y;
  switch (uMsg) {
    case WM_DESTROY:
      PostQuitMessage(0);
      exit(0);
      return 0;
    case WM_LBUTTONDOWN:
      gui->mouse_event(
          GUI::MouseEvent{GUI::MouseEvent::Type::press, gui->cursor_pos});
      break;
    case WM_LBUTTONUP:
      gui->mouse_event(
          GUI::MouseEvent{GUI::MouseEvent::Type::release, gui->cursor_pos});
      break;
    case WM_MOUSEMOVE:
      x = GET_X_LPARAM(lParam);
      y = GET_Y_LPARAM(lParam);
      gui->set_mouse_pos(x, gui->height - 1 - y);
      gui->mouse_event(
          GUI::MouseEvent{GUI::MouseEvent::Type::move, gui->cursor_pos});
      break;
    case WM_PAINT:
      break;
  }
  return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

TC_NAMESPACE_BEGIN

void GUI::process_event() {
  MSG msg;
  if (PeekMessage(&msg, hwnd, 0, 0, PM_REMOVE)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
}

void GUI::create_window() {
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
  data = (COLORREF *)calloc(width * height, sizeof(COLORREF));
  src = CreateCompatibleDC(hdc);
}

void GUI::redraw() {
  UpdateWindow(hwnd);
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

void GUI::set_title(std::string title) {
  SetWindowText(hwnd, title.c_str());
}

GUI::~GUI() {
  std::free(data);
  DeleteDC(src);
  gui_from_hwnd.erase(hwnd);
}

TC_NAMESPACE_END

#endif