/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
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
  TI_FORCE_INLINE static void run(Args &&... args) {
    F<bgn>::run(args...);
    RepeatFunctionHelper<F, bgn + 1, end, Args...>::run(
        std::forward<Args>(args)...);
  }
};

template <template <int> class F, int bgn, typename... Args>
struct RepeatFunctionHelper<F, bgn, bgn, Args...> {
  TI_FORCE_INLINE static void run(Args &&... args) {
    return;
  }
};

template <template <int> class F, int bgn, int end, typename... Args>
TI_FORCE_INLINE void repeat_function(Args &&... args) {
  RepeatFunctionHelper<F, bgn, end, Args...>::run(std::forward<Args>(args)...);
}
}  // namespace meta

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

template <typename... Args>
using type_switch_t = typename type_switch<Args...>::type;

template <typename T, typename G>
struct copy_refcv {
  TI_STATIC_ASSERT(
      (std::is_same<G, std::remove_cv_t<std::remove_reference_t<G>>>::value));
  static constexpr bool has_lvalue_ref = std::is_lvalue_reference<T>::value;
  static constexpr bool has_rvalue_ref = std::is_rvalue_reference<T>::value;
  static constexpr bool has_const =
      std::is_const<std::remove_reference_t<T>>::value;
  static constexpr bool has_volatile =
      std::is_volatile<std::remove_reference_t<T>>::value;
  using G1 = std::conditional_t<has_const, const G, G>;
  using G2 = std::conditional_t<has_volatile, volatile G1, G1>;
  using G3 = std::conditional_t<has_lvalue_ref, G2 &, G2>;
  using type = std::conditional_t<has_rvalue_ref, G3 &&, G3>;
};

template <typename T, typename G>
using copy_refcv_t = typename copy_refcv<T, G>::type;

template <class T, template <class...> class Template>
struct is_specialization : std::false_type {};

template <template <class...> class Template, class... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};

TI_STATIC_ASSERT((std::is_same<const volatile int, volatile const int>::value));
TI_STATIC_ASSERT(
    (std::is_same<int,
                  std::remove_volatile_t<
                      std::remove_const_t<const volatile int>>>::value));
TI_STATIC_ASSERT(
    (std::is_same<int,
                  std::remove_const_t<
                      std::remove_volatile_t<const volatile int>>>::value));
TI_STATIC_ASSERT((std::is_same<int &, std::add_const_t<int &>>::value));
TI_STATIC_ASSERT((std::is_same<copy_refcv_t<int, real>, real>::value));
TI_STATIC_ASSERT((std::is_same<copy_refcv_t<int &, real>, real &>::value));
TI_STATIC_ASSERT((copy_refcv<const int &, real>::has_lvalue_ref));
TI_STATIC_ASSERT(
    (std::is_same<copy_refcv<const int &, real>::G2, const real>::value));
TI_STATIC_ASSERT(
    (std::is_same<copy_refcv_t<const int &, real>, const real &>::value));
TI_STATIC_ASSERT((std::is_same<copy_refcv_t<const volatile int &, real>,
                               const volatile real &>::value));
TI_STATIC_ASSERT((is_specialization<std::vector<int>, std::vector>::value));

}  // namespace taichi
