#pragma once

#include <string>

#include "taichi/common/core.h"

#ifdef TI_PLATFORM_OSX

#include <objc/message.h>
#include <objc/objc.h>
#include <objc/runtime.h>

extern "C" {
void NSLog(/* NSString */ id format, ...);
}

namespace taichi {
namespace mac {

template <typename R, typename O, typename... Args>
R cast_call(O *i, const char *select, Args... args) {
  using func = R (*)(id, SEL, Args...);
  return ((func)(objc_msgSend))(reinterpret_cast<id>(i), sel_getUid(select),
                                args...);
}

template <typename O, typename... Args>
id call(O *i, const char *select, Args... args) {
  return cast_call<id>(i, select, args...);
}

template <typename R = id, typename... Args>
R clscall(const char *class_name, const char *select, Args... args) {
  using func = R (*)(id, SEL, Args...);
  return ((func)(objc_msgSend))((id)objc_getClass(class_name),
                                sel_getUid(select), args...);
}

template <typename O>
class NsObjDeleter {
 public:
  void operator()(O *o) {
    call(o, "release");
  }
};

template <typename O>
using nsobj_unique_ptr = std::unique_ptr<O, NsObjDeleter<O>>;

template <typename O>
nsobj_unique_ptr<O> wrap_as_nsobj_unique_ptr(O *nsobj) {
  return nsobj_unique_ptr<O>(nsobj);
}

template <typename O>
nsobj_unique_ptr<O> retain_and_wrap_as_nsobj_unique_ptr(O *nsobj) {
  // On creating an object, it could be either the caller or the callee's
  // responsibility to take the ownership of the object. By convention, method
  // names with "alloc", "new", "create" imply that the caller owns the object.
  // Otherwise, the object is tracked by an autoreleasepool before the callee
  // returns it.
  //
  // For an object that is owned by the callee (released by autoreleasepool), if
  // we want to *own* a reference to it, we must call [retain] to increment the
  // reference counting.
  //
  // In pratice, we find that each pthread (non main-thread) creates its own
  // autoreleasepool. Without retaining the object, it has caused double-free
  // on thread exit:
  // 1. nsobj_unique_ptr calls [release] in its destructor.
  // 2. autoreleasepool releases all the tracked objects upon thread exit.
  //
  // * https://stackoverflow.com/a/51080781/12003165
  // *
  // https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmRules.html#//apple_ref/doc/uid/20000994-SW1
  call(nsobj, "retain");
  return wrap_as_nsobj_unique_ptr(nsobj);
}

// Prepend "TI_" to native ObjC type names, otherwise clang-format thinks this
// is an ObjC file and is not happy formatting it.
struct TI_NSString;
struct TI_NSArray;

struct TI_NSRange {
  size_t location{0};
  size_t length{0};
};

// |str| must exist during the entire lifetime of the returned object, as it
// does not own the underlying memory. Think of it as std::string_view.
nsobj_unique_ptr<TI_NSString> wrap_string_as_ns_string(const std::string &str);

std::string to_string(TI_NSString *ns);

int ns_array_count(TI_NSArray *na);

template <typename R>
R ns_array_object_at_index(TI_NSArray *na, int i) {
  return cast_call<R>(na, "objectAtIndex:", i);
}

void ns_log_object(id obj);

struct TI_NSAutoreleasePool;

TI_NSAutoreleasePool *create_autorelease_pool();

void drain_autorelease_pool(TI_NSAutoreleasePool *pool);

class ScopedAutoreleasePool {
 public:
  ScopedAutoreleasePool();
  ~ScopedAutoreleasePool();

 private:
  TI_NSAutoreleasePool *pool_;
};

}  // namespace mac
}  // namespace taichi

#endif  // TI_PLATFORM_OSX
