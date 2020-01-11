#include <taichi/visual/gui.h>
#include <taichi/common/task.h>
#include <taichi/common/bit.h>

#if defined(TC_GUI_COCOA)

// https://stackoverflow.com/questions/4356441/mac-os-cocoa-draw-a-simple-pixel-on-a-canvas
// http://cocoadevcentral.com/d/intro_to_quartz/
// Modified based on
// https://github.com/CodaFi/C-Macs

// Obj-c runtime doc:
// https://developer.apple.com/documentation/objectivec/objective-c_runtime?language=objc

#include <objc/objc.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <objc/runtime.h>
#include <objc/message.h>
#include <objc/NSObjCRuntime.h>
#include <CoreGraphics/CGBase.h>
#include <CoreGraphics/CGGeometry.h>
#include <ApplicationServices/ApplicationServices.h>

template <typename C = id, typename... Args>
C call(id i, const char *select, Args... args) {
  using func = C (*)(id, SEL, Args...);
  return ((func)(objc_msgSend))(i, sel_getUid(select), args...);
}

template <typename C = id, typename... Args>
C call(const char *class_name, const char *select, Args... args) {
  using func = C (*)(id, SEL, Args...);
  return ((func)(objc_msgSend))((id)objc_getClass(class_name),
                                sel_getUid(select), args...);
}

extern id NSApp;
extern id const NSDefaultRunLoopMode;

typedef struct AppDel {
  Class isa;
  id window;
} AppDelegate;

class IdComparator {
 public:
  bool operator()(id a, id b) const {
    TC_STATIC_ASSERT(sizeof(a) == sizeof(taichi::int64));
    return taichi::bit::reinterpret_bits<taichi::int64>(a) <
           taichi::bit::reinterpret_bits<taichi::int64>(b);
  }
};

std::map<id, taichi::GUI *, IdComparator> gui_from_id;

enum {
  NSBorderlessWindowMask = 0,
  NSTitledWindowMask = 1 << 0,
  NSClosableWindowMask = 1 << 1,
  NSMiniaturizableWindowMask = 1 << 2,
  NSResizableWindowMask = 1 << 3,
};

Class ViewClass;

void redraw(id self, SEL _, CGRect __) {
  using namespace taichi;
  auto *gui = gui_from_id[self];
  auto width = gui->width, height = gui->height;
  auto &img = gui->canvas->img;
  auto &data = gui->img_data;
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      int index = 4 * (i + j * width);
      auto pixel = img[i][height - j - 1];
      data[index++] = uint8(clamp(int(pixel[0] * 255.0_f), 0, 255));
      data[index++] = uint8(clamp(int(pixel[1] * 255.0_f), 0, 255));
      data[index++] = uint8(clamp(int(pixel[2] * 255.0_f), 0, 255));
      data[index++] = 255;  // alpha
    }
  }

  CGDataProviderRef provider = CGDataProviderCreateWithData(
      nullptr, data.data(), gui->img_data_length, nullptr);
  CGColorSpaceRef colorspace = CGColorSpaceCreateDeviceRGB();
  CGImageRef image =
      CGImageCreate(width, height, 8, 32, width * 4, colorspace,
                    kCGBitmapByteOrder32Big | kCGImageAlphaPremultipliedLast,
                    provider, nullptr, true, kCGRenderingIntentDefault);
  CGContextRef context = call<CGContextRef>(
      call((id)objc_getClass("NSGraphicsContext"), "currentContext"),
      "graphicsPort");

  CGRect rect{{0, 0}, {CGFloat(width), CGFloat(height)}};
  CGContextDrawImage(context, rect, image);

  CGImageRelease(image);
  CGDataProviderRelease(provider);
  CGColorSpaceRelease(colorspace);
}

Class AppDelClass;
__attribute__((constructor)) static void initView() {
  ViewClass = objc_allocateClassPair((Class)objc_getClass("NSView"), "View", 0);
  // and again, we tell the runtime to add a function called -drawRect:
  // to our custom view. Note that there is an error in the type-specification
  // of this method, as I do not know the @encode sequence of 'CGRect' off
  // of the top of my head. As a result, there is a chance that the rect
  // parameter of the method may not get passed properly.
  class_addMethod(ViewClass, sel_getUid("drawRect:"), (IMP)redraw, "v@:");
  objc_registerClassPair(ViewClass);

  AppDelClass = objc_allocateClassPair((Class)objc_getClass("NSObject"),
                                       "AppDelegate", 0);
  objc_registerClassPair(AppDelClass);
}

TC_NAMESPACE_BEGIN

void GUI::create_window() {
  call("NSApplication", "sharedApplication");
  if (NSApp == nullptr) {
    fprintf(stderr, "Failed to initialized NSApplication.\nterminating.\n");
    return;
  }
  img_data_length = width * height * 4;
  img_data.resize(img_data_length);
  auto appDelObj = call("AppDelegate", "alloc");
  appDelObj = call(appDelObj, "init");
  call(NSApp, "setDelegate:", appDelObj);
  window = call("NSWindow", "alloc");
  auto rect = (CGRect){{0, 0}, {CGFloat(width), CGFloat(height)}};
  call(window, "initWithContentRect:styleMask:backing:defer:", rect,
       (NSTitledWindowMask | NSClosableWindowMask | NSResizableWindowMask |
        NSMiniaturizableWindowMask),
       0, false);
  view = call(call("View", "alloc"), "initWithFrame:", rect);
  gui_from_id[view] = this;
  call(window, "setContentView:", view);
  call(window, "becomeFirstResponder");
  call(window, "setAcceptsMouseMovedEvents:", YES);
  call(window, "makeKeyAndOrderFront:", window);
}

void GUI::process_event() {
  call(call("NSRunLoop", "currentRunLoop"),
       "runMode:beforeDate:", NSDefaultRunLoopMode,
       call("NSDate", "distantPast"));
  while (1) {
    auto event = call(
        NSApp, "nextEventMatchingMask:untilDate:inMode:dequeue:", NSUIntegerMax,
        call("NSDate", "distantPast"), NSDefaultRunLoopMode, YES);
    if (event != nullptr) {
      auto event_type = call<NSInteger>(event, "type");
      call(NSApp, "sendEvent:", event);
      call(NSApp, "updateWindows");
      auto p = call<CGPoint>(event, "locationInWindow");
      switch (event_type) {
        case 1:  // NSLeftMouseDown
          set_mouse_pos(p.x, p.y);
          mouse_event(MouseEvent{MouseEvent::Type::press, cursor_pos});
          break;
        case 2:  // NSLeftMouseUp
          set_mouse_pos(p.x, p.y);
          mouse_event(MouseEvent{MouseEvent::Type::release, cursor_pos});
          break;
        case 5:   // NSMouseMoved
        case 6:   // NSLeftMouseDragged
        case 7:   // NSRightMouseDragged
        case 27:  // NSNSOtherMouseDragged
          set_mouse_pos(p.x, p.y);
          mouse_event(MouseEvent{MouseEvent::Type::move, Vector2i(p.x, p.y)});
          break;
      }
    } else {
      break;
    }
  }
}

void GUI::set_title(std::string title) {
  auto str = call("NSString", "stringWithUTF8String:", title.c_str());
  call(window, "setTitle:", str);
  call(str, "release");
}

void GUI::redraw() {
  call(view, "setNeedsDisplay:", YES);
}

GUI::~GUI() {
}

TC_NAMESPACE_END

#endif