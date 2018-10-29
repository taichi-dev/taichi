#include <taichi/visual/gui.h>
#include <taichi/common/task.h>

#if defined(TC_GUI_COCOA)

#include <numeric>
// https://stackoverflow.com/questions/4356441/mac-os-cocoa-draw-a-simple-pixel-on-a-canvas
// http://cocoadevcentral.com/d/intro_to_quartz/
// Modified based on
// https://github.com/CodaFi/C-Macs

// Obj-c runtime doc:
// https://developer.apple.com/documentation/objectivec/objective-c_runtime?language=objc

#include <objc/message.h>
#include <objc/runtime.h>
#include <objc/objc.h>
#include <objc/runtime.h>
#include <objc/message.h>
#include <objc/NSObjCRuntime.h>
#include <CoreGraphics/CGBase.h>
#include <CoreGraphics/CGGeometry.h>
#include <ApplicationServices/ApplicationServices.h>

template <typename C=id, typename ... Args>
C call(id i, const char *select, Args ... args) {
  using func = C(id, SEL, Args...);
  return ((func *)(objc_msgSend))(i, sel_getUid(select), args...);
}

template <typename C=id, typename ... Args>
C call(const char *class_name, const char *select, Args ... args) {
  using func = C(id, SEL, Args...);
  return ((func *)(objc_msgSend))((id)objc_getClass(class_name), sel_getUid(select), args...);
}

extern id NSApp;
extern id const NSDefaultRunLoopMode;

static NSInteger width = 800;
static NSInteger height = 400;


typedef struct AppDel {
  Class isa;
  id window;
} AppDelegate;

enum {
  NSBorderlessWindowMask		= 0,
  NSTitledWindowMask			= 1 << 0,
  NSClosableWindowMask		= 1 << 1,
  NSMiniaturizableWindowMask	= 1 << 2,
  NSResizableWindowMask		= 1 << 3,
};

// This is a strong reference to the class of our custom view,
// In case we need it in the future.
Class ViewClass;

// This is a simple -drawRect implementation for our class. We could have
// used a NSTextField  or something of that sort instead, but I felt that this
// stuck with the C-based mentality of the application.

void redraw(id self, SEL _, CGRect __) {
  NSInteger dataLength = width * height * 4;
  UInt8 *data = (UInt8*)malloc(dataLength * sizeof(UInt8));
  static int t = 0;
  t++;
  for (int j=0; j < height; j++) {
    for (int i=0; i < width; i++) {
      int index = 4 * (i + j * width);
      data[index] = 255; //red
      data[++index] = (i + t * 100) % 255; //green
      data[++index] = 0;       //blue
      data[++index] = 255;     //alpha
    }
  }

  CGDataProviderRef provider = CGDataProviderCreateWithData(nullptr, data, dataLength, nullptr);
  CGColorSpaceRef colorspace = CGColorSpaceCreateDeviceRGB();
  CGImageRef image = CGImageCreate(width, height, 8, 32, width * 4, colorspace,
                                   kCGBitmapByteOrder32Big | kCGImageAlphaPremultipliedLast,
                                   provider, nullptr, true, kCGRenderingIntentDefault);

  CGContextRef context = call<CGContextRef>(call((id)objc_getClass("NSGraphicsContext"), "currentContext"), "graphicsPort");
  CGRect rect{{0, 0}, {CGFloat(width), CGFloat(height)}};
  CGContextDrawImage(context, rect, image);

  CGColorSpaceRelease(colorspace);
  CGDataProviderRelease(provider);
}

Class AppDelClass;

static void CreateAppDelegate() {
  AppDelClass = objc_allocateClassPair((Class)objc_getClass("NSObject"), "AppDelegate", 0);
  objc_registerClassPair(AppDelClass);
}

void RunApplication(void) {
  call("NSApplication", "sharedApplication");
  if (NSApp == nullptr) {
    fprintf(stderr,"Failed to initialized NSApplication.\nterminating.\n");
    return;
  }
  auto appDelObj = call("AppDelegate", "alloc");
  appDelObj = call(appDelObj, "init");
  call(NSApp, "setDelegate:", appDelObj);
  auto window = call("NSWindow", "alloc");
  auto rect = (CGRect){{0,0},{CGFloat(width),CGFloat(height)}};
  call(window, "initWithContentRect:styleMask:backing:defer:", rect, (NSTitledWindowMask | NSClosableWindowMask | NSResizableWindowMask | NSMiniaturizableWindowMask), 0, false);
  auto view = call(call("View", "alloc"), "initWithFrame:", rect);
  call(window, "setContentView:", view);
  call(window, "becomeFirstResponder");
  call(window, "makeKeyAndOrderFront:", window);

  //call(call((id)objc_getClass("NSRunLoop"), "currentRunLoop"), "run");
  call(call((id)objc_getClass("NSRunLoop"), "currentRunLoop"), "runMode:beforeDate:", NSDefaultRunLoopMode, call("NSDate", "distantPast"));
  for (;;) {
    printf("1\n");
    call(call((id)objc_getClass("NSRunLoop"), "currentRunLoop"), "runMode:beforeDate:", NSDefaultRunLoopMode, call("NSDate", "distantPast"));
    printf("2\n");
    auto event = call("NSApp",
                      "nextEventMatchingMask:untilDate:inMode:dequeue:",
                      NSUIntegerMax,
                      call("NSDate", "distantPast"),
                      NSDefaultRunLoopMode,
                      YES);
    call(view, "setNeedsDisplay:", YES);

    if (event != nullptr) {
      call("NSApp", "sendEvent:event", event);
    }
    // call("NSApp", "updateWindows");
  }
}

TC_NAMESPACE_BEGIN

GUI::GUI(const std::string &window_name, int width, int height, bool normalized_coord)
    : window_name(window_name),
      width(width),
      height(height),
      key_pressed(false) {

}

void GUI::redraw() {

}

void GUI::update() {

}

GUI::~GUI() {

}

auto test_cocoa_gui = []() {
  CreateAppDelegate();
  RunApplication();
};

TC_REGISTER_TASK(test_cocoa_gui);

TC_NAMESPACE_END


#endif