#include "objc_api.h"

#ifdef TI_PLATFORM_OSX

namespace taichi {
namespace mac {

nsobj_unique_ptr<TI_NSString> wrap_string_as_ns_string(const std::string &str) {
  constexpr int kNSUTF8StringEncoding = 4;
  id ns_string = clscall("NSString", "alloc");
  auto *ptr = cast_call<TI_NSString *>(
      ns_string,
      "initWithBytesNoCopy:length:encoding:freeWhenDone:", str.data(),
      str.size(), kNSUTF8StringEncoding, false);
  return wrap_as_nsobj_unique_ptr(ptr);
}

std::string to_string(TI_NSString *ns) {
  return cast_call<const char *>(ns, "UTF8String");
}

int ns_array_count(TI_NSArray *na) {
  return cast_call<int>(na, "count");
}

void ns_log_object(id obj) {
  // https://github.com/halide/Halide/blob/bce3abe95cf3aef36c6eb8dafb8c85c197408b6d/src/runtime/objc_support.h#L88-L92
  auto ns_str = wrap_string_as_ns_string("%@");
  NSLog(reinterpret_cast<id>(ns_str.get()), obj);
}

TI_NSAutoreleasePool *create_autorelease_pool() {
  return cast_call<TI_NSAutoreleasePool *>(
      clscall("NSAutoreleasePool", "alloc"), "init");
}

void drain_autorelease_pool(TI_NSAutoreleasePool *pool) {
  // "drain" is same as "release", so we don't need to release |pool| itself.
  // https://developer.apple.com/documentation/foundation/nsautoreleasepool
  call(pool, "drain");
}

ScopedAutoreleasePool::ScopedAutoreleasePool() {
  pool_ = create_autorelease_pool();
}

ScopedAutoreleasePool::~ScopedAutoreleasePool() {
  drain_autorelease_pool(pool_);
}

}  // namespace mac
}  // namespace taichi

#endif  // TI_PLATFORM_OSX
