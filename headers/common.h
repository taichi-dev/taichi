namespace taichi {
namespace Tlang {

struct Context {
  static constexpr int max_num_buffers = 16;
  static constexpr int max_num_parameters = 16;
  void *buffers[max_num_buffers];
  double parameters[max_num_parameters];

  template <typename T>
  T *get_buffer(int i) {
    return reinterpret_cast<T *>(buffers[i]);
  }

  template <typename T>
  T &get_parameter(int i) {
    return *reinterpret_cast<T *>(&parameters[i]);
  }
};

}
}

#if !defined(TC_INCLUDED)

#include <immintrin.h>
#include <cstdio>
using float32 = float;
using float64 = double;

#endif