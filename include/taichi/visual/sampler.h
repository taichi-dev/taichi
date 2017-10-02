/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/math.h>
#include <taichi/common/meta.h>
#include <vector>
#include <memory>
#include <string>

TC_NAMESPACE_BEGIN

class StateSequence {
 protected:
  int cursor = 0;

 public:
  virtual real sample() = 0;

  virtual real operator()() { return sample(); }

  int get_cursor() const { return cursor; }

  void assert_cursor_pos(int cursor) const {
    assert_info(
        this->cursor == cursor,
        std::string("Cursor position should be " + std::to_string(cursor) +
                    " instead of " + std::to_string(this->cursor)));
  }

  Vector2 next2() { return Vector2((*this)(), (*this)()); }

  Vector3 next3() { return Vector3((*this)(), (*this)(), (*this)()); }

  Vector4 next4() {
    return Vector4((*this)(), (*this)(), (*this)(), (*this)());
  }
};

class Sampler : public Unit {
 public:
  virtual real sample(int d, long long i) const = 0;
};
TC_INTERFACE(Sampler);

class RandomStateSequence : public StateSequence {
 private:
  std::shared_ptr<Sampler> sampler;
  long long instance;

 public:
  RandomStateSequence(std::shared_ptr<Sampler> sampler, long long instance)
      : sampler(sampler), instance(instance) {}

  real sample() override {
    assert_info(sampler != nullptr, "null sampler");
    real ret = sampler->sample(cursor++, instance);
    assert_info(ret >= 0, "sampler output should be non-neg");
    if (ret > 1 + 1e-5f) {
      printf("Warning: sampler returns value > 1: [%f]", ret);
    }
    if (ret >= 1) {
      ret = 0;
    }
    return ret;
  }
};

TC_NAMESPACE_END
