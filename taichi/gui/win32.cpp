#include "taichi/common/core.h"

#if defined(TI_PLATFORM_WINDOWS)
#include <windowsx.h>
#include "taichi/common/task.h"
#include "taichi/gui/gui.h"
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
        return std::string(1, key);
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
  int x, y;
  switch (uMsg) {
    case WM_DESTROY:
      PostQuitMessage(0);
      exit(0);
      return 0;
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
      x = GET_X_LPARAM(lParam);
      y = GET_Y_LPARAM(lParam);
      gui->set_mouse_pos(x, gui->height - 1 - y);
      gui->mouse_event(
          GUI::MouseEvent{GUI::MouseEvent::Type::move, gui->cursor_pos});
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
      exit(0);
      break;
  }
  return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

TI_NAMESPACE_BEGIN

void GUI::process_event() {
  MSG msg;
  if (PeekMessage(&msg, hwnd, 0, 0, PM_REMOVE)) {
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

  hwnd = CreateWindowEx(0,           // Optional window styles.
                        CLASS_NAME,  // Window class
                        std::wstring(window_name.begin(), window_name.end())
                            .data(),          // Window text
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
    TI_ERROR("Window creation failed");
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
  SetWindowText(hwnd, std::wstring(title.begin(), title.end()).data());
}

GUI::~GUI() {
  std::free(data);
  DeleteDC(src);
  gui_from_hwnd.erase(hwnd);
}

TI_NAMESPACE_END

#endif
