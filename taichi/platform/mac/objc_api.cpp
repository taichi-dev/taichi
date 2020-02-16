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

}  // namespace mac
}  // namespace taichi

#endif  // TI_PLATFORM_OSX
