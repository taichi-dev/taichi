#include <taichi/common/util.h>
#include <taichi/common/task.h>

// Note: some code is copied from MSDN:
// https://docs.microsoft.com/en-us/windows/desktop/learnwin32/introduction-to-windows-programming-in-c--

#include <windows.h>

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

int create_window() {
  // Register the window class.
  const char CLASS_NAME[] = "Sample Window Class";

  WNDCLASS wc = {};

  wc.lpfnWndProc = WindowProc;
  wc.hInstance = GetModuleHandle(0);
  wc.lpszClassName = CLASS_NAME;

  RegisterClass(&wc);
  
  // Create the window.

  HWND hwnd =
      CreateWindowEx(0,                           // Optional window styles.
                     CLASS_NAME,                  // Window class
                     "Learn to Program Windows",  // Window text
                     WS_OVERLAPPEDWINDOW,         // Window style

                     // Size and position
                     CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,

                     NULL,                // Parent window
                     NULL,                // Menu
                     GetModuleHandle(0),  // Instance handle
                     NULL                 // Additional application data
      );

  if (hwnd == NULL) {
    return 0;
  }

  ShowWindow(hwnd, SW_SHOWDEFAULT);

  // Run the message loop.

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
  switch (uMsg) {
    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;

    case WM_PAINT: {
      PAINTSTRUCT ps;
      HDC hdc = BeginPaint(hwnd, &ps);

      FillRect(hdc, &ps.rcPaint, (HBRUSH)(COLOR_WINDOW + 1));
      EndPaint(hwnd, &ps);
    }
      return 0;
  }
  return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

TC_NAMESPACE_BEGIN

auto win32guitest = []() {
  create_window();
};

TC_REGISTER_TASK(win32guitest);

TC_NAMESPACE_END