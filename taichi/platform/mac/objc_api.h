#include <string>

#include "taichi/common/core.h"

#ifdef TI_PLATFORM_OSX

#include <objc/message.h>
#include <objc/objc.h>
#include <objc/runtime.h>

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

// Prepend "TI_" to native ObjC type names, otherwise clang-format thinks this
// is an ObjC file and is not happy formatting it.
struct TI_NSString;

// |str| must exist during the entire lifetime of the returned object, as it
// does not own the underlying memory. Think of it as std::string_view.
nsobj_unique_ptr<TI_NSString> wrap_string_as_ns_string(const std::string &str);

}  // namespace mac
}  // namespace taichi

#endif  // TI_PLATFORM_OSX
