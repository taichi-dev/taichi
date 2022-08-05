#include "taichi/ui/gui/gui.h"

#include "taichi/common/task.h"
#include "taichi/util/bit.h"

#if defined(TI_GUI_COCOA)

#include <algorithm>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

#include "taichi/platform/mac/objc_api.h"

// https://stackoverflow.com/questions/4356441/mac-os-cocoa-draw-a-simple-pixel-on-a-canvas
// http://cocoadevcentral.com/d/intro_to_quartz/
// Modified based on
// https://github.com/CodaFi/C-Macs

// Obj-c runtime doc:
// https://developer.apple.com/documentation/objectivec/objective-c_runtime?language=objc

#include <ApplicationServices/ApplicationServices.h>
#include <Carbon/Carbon.h>
#include <CoreGraphics/CGBase.h>
#include <CoreGraphics/CGGeometry.h>
#include <objc/NSObjCRuntime.h>

namespace {
using taichi::mac::call;
using taichi::mac::cast_call;
using taichi::mac::clscall;

std::string str_tolower(std::string s) {
  // https://en.cppreference.com/w/cpp/string/byte/tolower
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

std::optional<std::string> try_get_alnum(ushort keycode) {
// Can someone tell me the reason why Apple didn't make these consecutive...
#define CASE(i) \
  { kVK_ANSI_##i, str_tolower(#i) }
  static const std::unordered_map<ushort, std::string> key2str = {
      CASE(0), CASE(1), CASE(2), CASE(3), CASE(4), CASE(5), CASE(6), CASE(7),
      CASE(8), CASE(9), CASE(A), CASE(B), CASE(C), CASE(D), CASE(E), CASE(F),
      CASE(G), CASE(H), CASE(I), CASE(J), CASE(K), CASE(L), CASE(M), CASE(N),
      CASE(O), CASE(P), CASE(Q), CASE(R), CASE(S), CASE(T), CASE(U), CASE(V),
      CASE(W), CASE(X), CASE(Y), CASE(Z),
  };
#undef CASE
  const auto iter = key2str.find(keycode);
  if (iter == key2str.end()) {
    return std::nullopt;
  }
  return iter->second;
}

std::optional<std::string> try_get_fnkey(ushort keycode) {
  // Or these...
#define STRINGIFY(x) #x
#define CASE(i) \
  { kVK_F##i, STRINGIFY(F##i) }
  static const std::unordered_map<ushort, std::string> key2str = {
      CASE(1),  CASE(2),  CASE(3),  CASE(4),  CASE(5),  CASE(6),
      CASE(7),  CASE(8),  CASE(9),  CASE(10), CASE(11), CASE(12),
      CASE(13), CASE(14), CASE(15), CASE(16),
  };
#undef CASE
#undef STRINGIFY
  const auto iter = key2str.find(keycode);
  if (iter == key2str.end()) {
    return std::nullopt;
  }
  return iter->second;
}

std::string lookup_keysym(ushort keycode) {
  // Full enum definition:
  // https://github.com/phracker/MacOSX-SDKs/blob/ef9fe35d5691b6dd383c8c46d867a499817a01b6/MacOSX10.6.sdk/System/Library/Frameworks/Carbon.framework/Versions/A/Frameworks/HIToolbox.framework/Versions/A/Headers/Events.h#L198-L315
  switch (keycode) {
    case kVK_LeftArrow:
      return "Left";
    case kVK_RightArrow:
      return "Right";
    case kVK_UpArrow:
      return "Up";
    case kVK_DownArrow:
      return "Down";
    case kVK_Tab:
      return "Tab";
    case kVK_Return:
      return "Return";
    // Mac Delete = Backspace on other platforms
    // Mac ForwardDelete (Fn + Delete) = Delete on other platforms
    case kVK_Delete:
      return "BackSpace";
    case kVK_Escape:
      return "Escape";
    case kVK_Space:
      return " ";
    default:
      break;
  }
  auto val_opt = try_get_alnum(keycode);
  if (val_opt.has_value()) {
    return *val_opt;
  }
  val_opt = try_get_fnkey(keycode);
  if (val_opt.has_value()) {
    return *val_opt;
  }
  return "Vk" + std::to_string((int)keycode);
}

// TODO(k-ye): Define all the magic numbers for Obj-C enums here
constexpr int NSApplicationActivationPolicyRegular = 0;
constexpr int NSEventTypeKeyDown = 10;
constexpr int NSEventTypeKeyUp = 11;
constexpr int NSEventTypeFlagsChanged = 12;
constexpr int NSEventTypeScrollWheel = 22;

struct ModifierFlagsHandler {
  struct Result {
    std::vector<std::string> released;
  };

  static Result handle(unsigned int flag,
                       std::unordered_map<std::string, bool> *active_flags) {
    constexpr int NSEventModifierFlagCapsLock = 1 << 16;
    constexpr int NSEventModifierFlagShift = 1 << 17;
    constexpr int NSEventModifierFlagControl = 1 << 18;
    constexpr int NSEventModifierFlagOption = 1 << 19;
    constexpr int NSEventModifierFlagCommand = 1 << 20;
    const static std::unordered_map<int, std::string> flag_mask_to_name = {
        {NSEventModifierFlagCapsLock, "Caps_Lock"},
        {NSEventModifierFlagShift, "Shift"},
        {NSEventModifierFlagControl, "Control"},
        // Mac Option = Alt on other platforms
        {NSEventModifierFlagOption, "Alt"},
        {NSEventModifierFlagCommand, "Command"},
    };
    Result result;
    for (const auto &kv : flag_mask_to_name) {
      bool &cur = (*active_flags)[kv.second];
      if (flag & kv.first) {
        cur = true;
      } else {
        if (cur) {
          // If previously pressed, trigger a release event
          result.released.push_back(kv.second);
        }
        cur = false;
      }
    }
    return result;
  }
};

// We need to give the View class a somewhat unique name, so that it won't
// conflict with other modules (e.g. matplotlib). See issue#998.
constexpr char kTaichiViewClassName[] = "TaichiGuiView";

}  // namespace

extern id NSApp;
extern id const NSDefaultRunLoopMode;

typedef struct AppDel {
  Class isa;
  id window;
} AppDelegate;

class IdComparator {
 public:
  bool operator()(id a, id b) const {
    TI_STATIC_ASSERT(sizeof(a) == sizeof(taichi::int64));
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

void updateLayer(id self, SEL _) {
  using namespace taichi;
  auto *gui = gui_from_id[self];
  auto width = gui->width, height = gui->height;
  uint8_t *data_ptr = nullptr;
  if (gui->fast_gui) {
    data_ptr = reinterpret_cast<uint8_t *>(gui->fast_buf);
  } else {
    auto &img = gui->canvas->img;
    auto &data = gui->img_data;
    data_ptr = data.data();
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
  }

  CGDataProviderRef provider = CGDataProviderCreateWithData(
      nullptr, data_ptr, gui->img_data_length, nullptr);
  CGColorSpaceRef colorspace = CGColorSpaceCreateDeviceRGB();
  CGImageRef image =
      CGImageCreate(width, height, 8, 32, width * 4, colorspace,
                    kCGBitmapByteOrder32Big | kCGImageAlphaPremultipliedLast,
                    provider, nullptr, true, kCGRenderingIntentDefault);
  // Profiling showed that CGContextDrawImage can be rather slow (~50ms per
  // frame!), so we instead set the image as the content of the view's layer.
  // See also:
  // * slow CGContextDrawImage: https://stackoverflow.com/a/7599794/12003165
  // * CALayer + CGImage: https://stackoverflow.com/a/48310419/12003165
  // * profiling:
  // https://github.com/taichi-dev/taichi/issues/489#issuecomment-589955458
  call(call(gui->view, "layer"), "setContents:", image);
  CGImageRelease(image);
  CGDataProviderRelease(provider);
  CGColorSpaceRelease(colorspace);
}

BOOL windowShouldClose(id self, SEL _, id sender) {
  auto *gui = gui_from_id[sender];
  gui->window_received_close.store(true);
  return true;
}

Class ViewClass;
Class AppDelClass;

__attribute__((constructor)) static void initView() {
  ViewClass = objc_allocateClassPair((Class)objc_getClass("NSView"),
                                     kTaichiViewClassName, 0);
  // There are two ways to update NSView's content, either via "drawRect:" or
  // "updateLayer". Updating via layer can be a lot faster, so we use this
  // method. See also:
  // https://developer.apple.com/documentation/appkit/nsview/1483461-wantsupdatelayer?language=objc
  // https://stackoverflow.com/a/51899686/12003165
  //
  // Also, it should be noted that if NSView's layer is enabled (via
  // [view setWantsLyaer:YES]), but "drawRect:" is used, then the content is
  // drawn and cleared rapidly, causing a flickering screen. It seems that the
  // view itself and the underlying layer were overwriting each other's content.
  // https://stackoverflow.com/a/11321521/12003165
  class_addMethod(ViewClass, sel_getUid("updateLayer" /* no colon */),
                  (IMP)updateLayer, "v@:");
  objc_registerClassPair(ViewClass);

  AppDelClass = objc_allocateClassPair((Class)objc_getClass("NSObject"),
                                       "AppDelegate", 0);
  Protocol *WinDelProtocol = objc_getProtocol("NSWindowDelegate");
  class_addMethod(AppDelClass, sel_getUid("windowShouldClose:"),
                  (IMP)windowShouldClose, "c@:@");
  class_addProtocol(AppDelClass, WinDelProtocol);
  objc_registerClassPair(AppDelClass);
}

TI_NAMESPACE_BEGIN

void GUI::create_window() {
  clscall("NSApplication", "sharedApplication");
  if (NSApp == nullptr) {
    fprintf(stderr, "Failed to initialized NSApplication.\nterminating.\n");
    return;
  }
  // I finally found how to bring the NSWindow to the front and to handle
  // keyboard events in these posts:
  // https://stackoverflow.com/a/11010614/12003165
  // http://www.cocoawithlove.com/2010/09/minimalist-cocoa-programming.html
  //
  // The problem was that, a Cocoa app without NIB files (app bundle,
  // info.plist, whatever the meta files are) by default has a policy of
  // NSApplicationActivationPolicyProhibited.
  // (https://developer.apple.com/documentation/appkit/nsapplicationactivationpolicy/nsapplicationactivationpolicyprohibited?language=objc)
  call(NSApp, "setActivationPolicy:", NSApplicationActivationPolicyRegular);
  // This doesn't seem necessary, but in case there's some weird bug causing the
  // Window not to be brought to the front, try enable this.
  // https://stackoverflow.com/a/7460187/12003165
  // call(NSApp, "activateIgnoringOtherApps:", YES);
  img_data_length = width * height * 4;
  img_data.resize(img_data_length);
  auto appDelObj = clscall("AppDelegate", "alloc");
  appDelObj = call(appDelObj, "init");
  call(NSApp, "setDelegate:", appDelObj);
  window = clscall("NSWindow", "alloc");
  auto rect = (CGRect){{0, 0}, {CGFloat(width), CGFloat(height)}};
  call(window, "initWithContentRect:styleMask:backing:defer:", rect,
       (NSTitledWindowMask | NSClosableWindowMask | NSResizableWindowMask |
        NSMiniaturizableWindowMask),
       0, false);
  view = call(clscall(kTaichiViewClassName, "alloc"), "initWithFrame:", rect);
  gui_from_id[view] = this;
  // Needed by NSWindowDelegate
  gui_from_id[window] = this;
  // Use layer to speed up the draw
  // https://developer.apple.com/documentation/appkit/nsview/1483695-wantslayer?language=objc
  call(view, "setWantsLayer:", YES);
  call(window, "setDelegate:", appDelObj);
  call(window, "setContentView:", view);
  call(window, "becomeFirstResponder");
  call(window, "setAcceptsMouseMovedEvents:", YES);
  call(window, "makeKeyAndOrderFront:", window);
  if (fullscreen) {
    call(window, "toggleFullScreen:");
  }
}

void GUI::process_event() {
  call(clscall("NSRunLoop", "currentRunLoop"),
       "runMode:beforeDate:", NSDefaultRunLoopMode,
       clscall("NSDate", "distantPast"));
  while (1) {
    auto event = call(
        NSApp, "nextEventMatchingMask:untilDate:inMode:dequeue:", NSUIntegerMax,
        clscall("NSDate", "distantPast"), NSDefaultRunLoopMode, YES);
    if (event != nullptr) {
      auto event_type = cast_call<NSInteger>(event, "type");
      call(NSApp, "sendEvent:", event);
      call(NSApp, "updateWindows");
      auto p = cast_call<CGPoint>(event, "locationInWindow");
      ushort keycode = 0;
      std::string keysym;
      switch (event_type) {
        case 1:  // NSLeftMouseDown
          set_mouse_pos(p.x, p.y);
          mouse_event(MouseEvent{MouseEvent::Type::press, cursor_pos});
          key_events.push_back(
              GUI::KeyEvent{GUI::KeyEvent::Type::press, "LMB", cursor_pos});
          break;
        case 2:  // NSLeftMouseUp
          set_mouse_pos(p.x, p.y);
          mouse_event(MouseEvent{MouseEvent::Type::release, cursor_pos});
          key_events.push_back(
              GUI::KeyEvent{GUI::KeyEvent::Type::release, "LMB", cursor_pos});
          break;
        case 3:  // NSEventTypeRightMouseDown
          key_events.push_back(
              GUI::KeyEvent{GUI::KeyEvent::Type::press, "RMB", cursor_pos});
          break;
        case 4:  // NSEventTypeRightMouseUp
          key_events.push_back(
              GUI::KeyEvent{GUI::KeyEvent::Type::release, "RMB", cursor_pos});
          break;
        case 5:   // NSMouseMoved
        case 6:   // NSLeftMouseDragged
        case 7:   // NSRightMouseDragged
        case 27:  // NSNSOtherMouseDragged
          set_mouse_pos(p.x, p.y);
          key_events.push_back(
              GUI::KeyEvent{GUI::KeyEvent::Type::move, "Motion", cursor_pos});
          mouse_event(MouseEvent{MouseEvent::Type::move, Vector2i(p.x, p.y)});
          break;
        case NSEventTypeKeyDown:
        case NSEventTypeKeyUp: {
          keycode = cast_call<ushort>(event, "keyCode");
          keysym = lookup_keysym(keycode);
          auto kev_type = (event_type == NSEventTypeKeyDown)
                              ? KeyEvent::Type::press
                              : KeyEvent::Type::release;
          key_events.push_back(KeyEvent{kev_type, keysym, cursor_pos});
          break;
        }
        case NSEventTypeFlagsChanged: {
          const auto modflag = cast_call<unsigned long>(event, "modifierFlags");
          const auto r =
              ModifierFlagsHandler::handle(modflag, &active_modifier_flags);
          for (const auto &key : r.released) {
            key_events.push_back(
                KeyEvent{KeyEvent::Type::release, key, cursor_pos});
          }
          break;
        }
        case NSEventTypeScrollWheel: {
          set_mouse_pos(p.x, p.y);
          const auto dx = (int)cast_call<CGFloat>(event, "scrollingDeltaX");
          // Mac trackpad's vertical scroll is reversed.
          const auto dy = -(int)cast_call<CGFloat>(event, "scrollingDeltaY");
          key_events.push_back(KeyEvent{KeyEvent::Type::move, "Wheel",
                                        cursor_pos, Vector2i{dx, dy}});
          break;
        }
      }
    } else {
      break;
    }
  }

  for (const auto &kv : active_modifier_flags) {
    if (kv.second) {
      key_events.push_back(
          KeyEvent{KeyEvent::Type::press, kv.first, cursor_pos});
    }
  }
  if (window_received_close.load()) {
    send_window_close_message();
    window_received_close.store(false);
  }
}

void GUI::set_title(std::string title) {
  auto str = clscall("NSString", "stringWithUTF8String:", title.c_str());
  call(window, "setTitle:", str);
  call(str, "release");
}

void GUI::redraw() {
  call(view, "setNeedsDisplay:", YES);
}

GUI::~GUI() {
  if (show_gui) {
    call(window, "close");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    process_event();
  }
}

TI_NAMESPACE_END

#endif
