#include <string>

namespace taichi {
namespace Tlang {

struct Context {
  using Buffer = void *;
  Buffer buffers[1];

  Context() {
    for (int i = 0; i < 1; i++)
      buffers[i] = nullptr;
  }

  Context(void *x) : Context() {
    buffers[0] = x;
  }
};

}  // namespace Tlang

};  // namespace taichi
