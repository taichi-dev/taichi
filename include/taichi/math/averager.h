/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <functional>

#include <taichi/math/math.h>

TC_NAMESPACE_BEGIN

class RunningAverage {
 private:
  double total_value, total_weight;
  real safe_value;

 public:
  RunningAverage() {
    safe_value = 0.0_f;
    clear();
  }

  void insert(real value, real weight) {
    total_value += value;
    total_weight += weight;
  }

  real get_average() {
    if (total_weight == 0) {
      return safe_value;
    }
    return max(real(total_value / total_weight), safe_value);
  }

  void clear() {
    total_value = 0.0;
    total_weight = 0.0;
  }

  void set_safe_value(real safe_value) {
    this->safe_value = safe_value;
  }
};

TC_NAMESPACE_END
