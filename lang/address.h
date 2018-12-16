#pragma once

#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <dlfcn.h>
#include <set>
#include "../headers/common.h"

TC_NAMESPACE_BEGIN
namespace Tlang {

template <typename T>
using Handle = std::shared_ptr<T>;

class Expr;

struct Address {
  int64 buffer_id;
  int64 coeff_i;
  int64 coeff_imax;
  int64 coeff_const;  // offset

  // AOSOA: i / a * b
  int64 coeff_aosoa_group_size;
  int64 coeff_aosoa_stride;

  TC_IO_DEF(buffer_id,
            coeff_i,
            coeff_imax,
            coeff_const,
            coeff_aosoa_stride,
            coeff_aosoa_group_size);

  Address() {
    buffer_id = -1;
    coeff_i = 0;
    coeff_imax = 0;
    coeff_const = 0;
    coeff_aosoa_group_size = 0;
    coeff_aosoa_stride = 0;
  }

  bool initialized() {
    return buffer_id != -1;
  }

  TC_FORCE_INLINE bool same_type(Address o) {
    return buffer_id == o.buffer_id && coeff_i == o.coeff_i &&
           coeff_imax == o.coeff_imax &&
           coeff_aosoa_group_size == o.coeff_aosoa_group_size &&
           coeff_aosoa_stride == o.coeff_aosoa_stride;
  }

  TC_FORCE_INLINE bool operator==(Address o) {
    return buffer_id == o.buffer_id && coeff_i == o.coeff_i &&
           coeff_imax == o.coeff_imax && coeff_const == o.coeff_const &&
           coeff_aosoa_group_size == o.coeff_aosoa_group_size &&
           coeff_aosoa_stride == o.coeff_aosoa_group_size;
  }

  TC_FORCE_INLINE int64 offset() {
    return coeff_const;
  }

  int64 eval(int64 i, int64 n) {
    TC_ASSERT(initialized());
    if (coeff_aosoa_stride != 0) {
      return coeff_i * i + coeff_imax * n + coeff_const +
             (i / coeff_aosoa_group_size) * coeff_aosoa_stride;
    } else {
      return coeff_i * i + coeff_imax * n + coeff_const;
    }
  }
};

}

TC_NAMESPACE_END
