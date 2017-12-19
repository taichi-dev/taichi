/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <cstring>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <iostream>

namespace taichi {

namespace meta {
template <template <int> class F, int bgn, int end, typename... Args>
struct RepeatFunctionHelper {
  TC_FORCE_INLINE static void run(Args &&... args) {
    F<bgn>::run(args...);
    RepeatFunctionHelper<F, bgn + 1, end, Args...>::run(
        std::forward<Args>(args)...);
  }
};

template <template <int> class F, int bgn, typename... Args>
struct RepeatFunctionHelper<F, bgn, bgn, Args...> {
  TC_FORCE_INLINE static void run(Args &&... args) {
    return;
  }
};

template <template <int> class F, int bgn, int end, typename... Args>
TC_FORCE_INLINE void repeat_function(Args &&... args) {
  RepeatFunctionHelper<F, bgn, end, Args...>::run(std::forward<Args>(args)...);
}
}

using meta::repeat_function;

template <typename option, typename... Args>
struct type_switch {
  using type = typename std::conditional<
      std::is_same<typename option::first_type, std::true_type>::value,
      typename option::second_type,
      typename type_switch<Args...>::type>::type;
};

template <typename option>
struct type_switch<option> {
  static_assert(
      std::is_same<typename option::first_type, std::true_type>::value,
      "None of the options in type_switch works.");
  using type = typename option::second_type;
};

namespace STATIC_IF {
// reference: https://github.com/wichtounet/cpp_utils

struct identity {
  template <typename T>
  T operator()(T &&x) const {
    return std::forward<T>(x);
  }
};

template <bool Cond>
struct statement {
  template <typename F>
  void then(const F &f) {
    f(identity());
  }

  template <typename F>
  void else_(const F &) {
  }
};

template <>
struct statement<false> {
  template <typename F>
  void then(const F &) {
  }

  template <typename F>
  void else_(const F &f) {
    f(identity());
  }
};

template <bool Cond, typename F>
inline statement<Cond> static_if(F const &f) {
  statement<Cond> if_;
  if_.then(f);
  return if_;
}
}

using STATIC_IF::static_if;

#define TC_STATIC_IF(x) taichi::static_if<(x)>([&](const auto& _____) -> void {
#define TC_STATIC_ELSE \
  }).else_([&](const auto &_____) -> void {
#define TC_STATIC_END_IF \
  });

// After we switch to C++17, we should use
// (Note the the behaviour of 'return' is still different.)

/*
#define TC_STATIC_IF(x) if constexpr(x) {
#define TC_STATIC_ELSE \
    } else {
#define TC_STATIC_END_IF \
    }
*/
}
