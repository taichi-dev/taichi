#include "taichi/ui/gui/gui.h"

// GGUI is not supported on Android as the window management is handled by the
// framework directly. It also provides a Canvas through Skia library that users
// can leverage for rendering of 2D elements (circle, rectangle, ...)
#if defined(TI_GUI_ANDROID)
#include <cstdlib>

TI_NAMESPACE_BEGIN

void GUI::process_event() {
  TI_ERROR("GGUI not supported on Android");
}

void GUI::create_window() {
  TI_ERROR("GGUI not supported on Android");
}

void GUI::redraw() {
  TI_ERROR("GGUI not supported on Android");
}

void GUI::set_title(std::string title) {
  TI_ERROR("GGUI not supported on Android");
}

GUI::~GUI() {
}

TI_NAMESPACE_END

#endif
