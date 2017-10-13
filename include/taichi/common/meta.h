/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/config.h>
#include <taichi/common/asset_manager.h>
#include <cstring>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <iostream>

TC_NAMESPACE_BEGIN

namespace meta {
template <template <int> typename F, int bgn, int end, typename... Args>
struct RepeatFunctionHelper {
  TC_FORCE_INLINE static void run(Args &&... args) {
    F<bgn>::run(args...);
    RepeatFunctionHelper<F, bgn + 1, end, Args...>::run(
        std::forward<Args>(args)...);
  }
};

template <template <int> typename F, int bgn, typename... Args>
struct RepeatFunctionHelper<F, bgn, bgn, Args...> {
  TC_FORCE_INLINE static void run(Args &&... args) {
    return;
  }
};

template <template <int> typename F, int bgn, int end, typename... Args>
TC_FORCE_INLINE void repeat_function(Args &&... args) {
  RepeatFunctionHelper<F, bgn, end, Args...>::run(std::forward<Args>(args)...);
}
}

using meta::repeat_function;

TC_NAMESPACE_END
