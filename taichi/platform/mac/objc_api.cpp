#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "objc_api.h"

#ifdef TI_PLATFORM_OSX

namespace taichi {
namespace mac {

nsobj_unique_ptr<TI_NSString> wrap_string_as_ns_string(const std::string &str) {
  return wrap_as_nsobj_unique_ptr(
      NS::String::string(str.c_str(), NS::StringEncoding::UTF8StringEncoding));
}

std::string to_string(TI_NSString *ns) {
  // (penguinliong) Specify length to avoid crashes on `nil`.
  return std::string(ns->utf8String(),
                     ns->lengthOfBytes(NS::StringEncoding::UTF8StringEncoding));
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
