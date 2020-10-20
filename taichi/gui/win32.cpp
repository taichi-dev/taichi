#include "taichi/common/core.h"

#if defined(TI_PLATFORM_WINDOWS)
#include <windowsx.h>
#include "taichi/common/task.h"
#include "taichi/gui/gui.h"
#include <cctype>
#include <map>

// Note: some code is copied from MSDN:
// https://docs.microsoft.com/en-us/windows/desktop/learnwin32/introduction-to-windows-programming-in-c--

std::map<HWND, taichi::GUI *> gui_from_hwnd;

static std::string lookup_keysym(WPARAM wParam, LPARAM lParam) {
  int key = wParam;
  switch (key) {
    /*** http://kbdedit.com/manual/low_level_vk_list.html ***/
    case VK_LEFT:
      return "Left";
    case VK_RIGHT:
      return "Right";
    case VK_UP:
      return "Up";
    case VK_DOWN:
      return "Down";
    case VK_TAB:
      return "Tab";
    case VK_RETURN:
      return "Return";
    case VK_BACK:
      return "BackSpace";
    case VK_ESCAPE:
      return "Escape";
    case VK_SHIFT:
    case VK_LSHIFT:
      return "Shift_L";
    case VK_RSHIFT:
      return "Shift_R";
    case VK_CONTROL:
    case VK_LCONTROL:
      return "Control_L";
    case VK_RCONTROL:
      return "Control_R";
    case VK_MENU:
    case VK_LMENU:
      return "Alt_L";
    case VK_RMENU:
      return "Alt_R";
    case VK_CAPITAL:
      return "Caps_Lock";
    /*** TODO: win32 keyboard WIP, add more cases, match XKeysymToString() ***/
    default:
      if (VK_F1 <= key && key <= VK_F12)
        return fmt::format("F{}", key - VK_F1);
      else if (isascii(key))
        return std::string(1, std::tolower(key));
      else
        return fmt::format("Vk{}", key);
  }
}

LRESULT CALLBACK WindowProc(HWND hwnd,
                            UINT uMsg,
                            WPARAM wParam,
                            LPARAM lParam) {
  auto dc = GetDC(hwnd);
  auto gui = gui_from_hwnd[hwnd];
  using namespace taichi;
  POINT p{};
  switch (uMsg) {
    case WM_LBUTTONDOWN:
      gui->mouse_event(
          GUI::MouseEvent{GUI::MouseEvent::Type::press, gui->cursor_pos});
      gui->key_events.push_back(
          GUI::KeyEvent{GUI::KeyEvent::Type::press, "LMB", gui->cursor_pos});
      break;
    case WM_LBUTTONUP:
      gui->mouse_event(
          GUI::MouseEvent{GUI::MouseEvent::Type::release, gui->cursor_pos});
      gui->key_events.push_back(
          GUI::KeyEvent{GUI::KeyEvent::Type::release, "LMB", gui->cursor_pos});
      break;
    case WM_MBUTTONDOWN:
      gui->key_events.push_back(
          GUI::KeyEvent{GUI::KeyEvent::Type::press, "MMB", gui->cursor_pos});
      break;
    case WM_MBUTTONUP:
      gui->key_events.push_back(
          GUI::KeyEvent{GUI::KeyEvent::Type::release, "MMB", gui->cursor_pos});
      break;
    case WM_RBUTTONDOWN:
      gui->key_events.push_back(
          GUI::KeyEvent{GUI::KeyEvent::Type::press, "RMB", gui->cursor_pos});
      break;
    case WM_RBUTTONUP:
      gui->key_events.push_back(
          GUI::KeyEvent{GUI::KeyEvent::Type::release, "RMB", gui->cursor_pos});
      break;
    case WM_MOUSEMOVE:
      p = {GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
      gui->set_mouse_pos(p.x, gui->height - 1 - p.y);
      gui->mouse_event(
          GUI::MouseEvent{GUI::MouseEvent::Type::move, gui->cursor_pos});
      break;
    case WM_MOUSEWHEEL:
      p = {GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
      ScreenToClient(hwnd, &p);
      gui->set_mouse_pos(p.x, gui->height - 1 - p.y);
      gui->key_events.push_back(
          GUI::KeyEvent{GUI::KeyEvent::Type::move, "Wheel", gui->cursor_pos,
                        Vector2i{0, GET_WHEEL_DELTA_WPARAM(wParam)}});
      break;
    case WM_MOUSEHWHEEL:
      p = {GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
      ScreenToClient(hwnd, &p);
      gui->set_mouse_pos(p.x, gui->height - 1 - p.y);
      gui->key_events.push_back(
          GUI::KeyEvent{GUI::KeyEvent::Type::move, "Wheel", gui->cursor_pos,
                        Vector2i{GET_WHEEL_DELTA_WPARAM(wParam), 0}});
      break;
    case WM_PAINT:
      break;
    case WM_KEYDOWN:
      gui->key_pressed = true;
      gui->key_events.push_back(GUI::KeyEvent{GUI::KeyEvent::Type::press,
                                              lookup_keysym(wParam, lParam),
                                              gui->cursor_pos});
      break;
    case WM_KEYUP:
      gui->key_events.push_back(GUI::KeyEvent{GUI::KeyEvent::Type::release,
                                              lookup_keysym(wParam, lParam),
                                              gui->cursor_pos});
      break;
    case WM_CLOSE:
      // https://stackoverflow.com/questions/3155782/what-is-the-difference-between-wm-quit-wm-close-and-wm-destroy-in-a-windows-pr
      gui->send_window_close_message();
      break;
  }
  return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

TI_NAMESPACE_BEGIN

void GUI::process_event() {
  MSG msg;
  if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
}

void GUI::create_window() {
  auto CLASS_NAME = L"Taichi Win32 Window";

  WNDCLASS wc = {};

  wc.lpfnWndProc = WindowProc;
  wc.hInstance = GetModuleHandle(0);
  wc.lpszClassName = CLASS_NAME;

  RegisterClass(&wc);

  RECT window_rect;
  window_rect.left = 0;
  window_rect.right = width;
  window_rect.top = 0;
  window_rect.bottom = height;

  AdjustWindowRect(&window_rect, WS_OVERLAPPEDWINDOW, false);

  hwnd = CreateWindowEx(0,           // Optional window styles.
                        CLASS_NAME,  // Window class
                        std::wstring(window_name.begin(), window_name.end())
                            .data(),          // Window text
                        WS_OVERLAPPEDWINDOW,  // Window style
                        // Size and position
                        CW_USEDEFAULT, CW_USEDEFAULT,
                        window_rect.right - window_rect.left,
                        window_rect.bottom - window_rect.top,
                        NULL,                // Parent window
                        NULL,                // Menu
                        GetModuleHandle(0),  // Instance handle
                        NULL                 // Additional application data
  );
  TI_ERROR_IF(hwnd == NULL, "Window creation failed");
  gui_from_hwnd[hwnd] = this;

  if (fullscreen) {
    // https://www.cnblogs.com/lidabo/archive/2012/07/17/2595452.html
    LONG style = GetWindowLong(hwnd, GWL_STYLE);
    style &= ~WS_CAPTION & ~WS_SIZEBOX;
    SetWindowLong(hwnd, GWL_STYLE, style);
    SetWindowPos(hwnd, NULL, 0, 0, GetSystemMetrics(SM_CXSCREEN),
                 GetSystemMetrics(SM_CYSCREEN), SWP_NOZORDER);
  }

  ShowWindow(hwnd, SW_SHOWDEFAULT);
  hdc = GetDC(hwnd);
  data = (COLORREF *)calloc(width * height, sizeof(COLORREF));
  src = CreateCompatibleDC(hdc);
}

void GUI::redraw() {
  UpdateWindow(hwnd);
  if (!fast_gui) {
    // http://www.cplusplus.com/reference/cstdlib/calloc/
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
        auto c = reinterpret_cast<unsigned char *>(data + (j * width) + i);
        auto d = canvas->img[i][height - j - 1];
        c[0] = uint8(clamp(int(d[2] * 255.0_f), 0, 255));
        c[1] = uint8(clamp(int(d[1] * 255.0_f), 0, 255));
        c[2] = uint8(clamp(int(d[0] * 255.0_f), 0, 255));
        c[3] = 0;
      }
    }
  }
  bitmap = CreateBitmap(width, height, 1, 32,
                        fast_gui ? (void *)fast_buf : (void *)data);
  SelectObject(src, bitmap);
  BitBlt(hdc, 0, 0, width, height, src, 0, 0, SRCCOPY);
  DeleteObject(bitmap);
}

void GUI::set_title(std::string title) {
  SetWindowText(hwnd, std::wstring(title.begin(), title.end()).data());
}

GUI::~GUI() {
  if (show_gui) {
    std::free(data);
    DeleteDC(src);
    DestroyWindow(hwnd);
    gui_from_hwnd.erase(hwnd);
  }
}

TI_NAMESPACE_END

#endif
