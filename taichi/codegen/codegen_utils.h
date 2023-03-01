#pragma once
#include "taichi/program/program.h"
#include <regex>

namespace taichi::lang {

inline bool codegen_vector_type(const CompileConfig &config) {
  return !config.real_matrix_scalarize;
}

// Merge the printf specifiers from user-defined one and taichi data types, and
// rewrite the specifiers accroding to backend capabbilities.
inline std::string merge_printf_specifier(
    std::optional<std::string> const &from_user,
    std::string const &from_data_type,
    Arch arch) {
  if (!from_user.has_value()) {
    return from_data_type;
  }
  std::string const &user = from_user.value();

  // printf format string specifiers:
  // %[flags]+[width][.precision][length][conversion]
  // https://en.cppreference.com/w/cpp/io/c/fprintf
  const std::regex user_re = std::regex(
      "([-+ #0]+)?"
      "(\\d+|\\*)?"
      "(\\.(?:\\d+|\\*))?"
      "([hljztL]|hh|ll)?"
      "([csdioxXufFeEaAgGnp])?");
  std::smatch user_match;
  bool user_matched = std::regex_match(user, user_match, user_re);
  if (user_matched == false) {
    TI_ERROR("{} is not a valid printf specifier.", user)
  }
  std::string user_flags = user_match[1];
  std::string user_width = user_match[2];
  std::string user_precision = user_match[3];
  std::string user_length = user_match[4];
  std::string user_conversion = user_match[5];

  if (user_width == "*" || user_precision == ".*" || user_conversion == "n") {
    TI_ERROR("The {} printf specifier is not supported", user)
  }

  const std::regex dt_re = std::regex(
      "%"
      "(\\.(?:\\d+))?"
      "([hljztL]|hh|ll)?"
      "([csdioxXufFeEaAgGnp])?");
  std::smatch dt_match;
  bool dt_matched = std::regex_match(from_data_type, dt_match, dt_re);
  TI_ASSERT(dt_matched);
  std::string dt_precision = dt_match[1];
  std::string dt_length = dt_match[2];
  std::string dt_conversion = dt_match[3];

  // Constant for convensions in group.
  constexpr std::string_view signed_group = "di";
  constexpr std::string_view unsigned_group = "oxXu";
  constexpr std::string_view float_group = "fFeEaAgG";

  // Vulkan doesn't support length, flags, or width specifier.
  // https://vulkan.lunarg.com/doc/view/1.2.162.1/linux/debug_printf.html
  //
  // CUDA supports all of them.
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#format-specifiers
  if (arch == Arch::vulkan) {
    if (!user_flags.empty()) {
      TI_WARN(
          "The printf flags '{}' are not supported in Vulkan, "
          "and will be discarded.",
          user_flags);
      user_flags.clear();
    }
    if (!user_width.empty()) {
      TI_WARN(
          "The printf width modifier '{}' is not supported in Vulkan, "
          "and will be discarded.",
          user_width);
      user_width.clear();
    }
    // except for unsigned long
    if (!user_length.empty() &&
        !(user_length == "l" && !user_conversion.empty() &&
          unsigned_group.find(user_conversion) != std::string::npos)) {
      TI_WARN(
          "The printf length modifier '{}' is not supported in Vulkan, "
          "and will be discarded.",
          user_length);
      user_length.clear();
    }
    if (dt_precision == ".12" || dt_length == "ll") {
      TI_WARN(
          "Vulkan does not support 64-bit printing, except for unsigned long.");
    }
  }

  // Replace user_precision with dt_precision if the former is empty,
  // otherwise use user specified precision.
  if (user_precision.empty()) {
    user_precision = dt_precision;
  }

  // Discard user_length and give warning if it doesn't match with dt_length.
  if (user_length != dt_length) {
    if (!user_length.empty()) {
      TI_WARN("The printf length specifier '{}' is overritten by '{}'",
              user_length, dt_length);
    }
    user_length = dt_length;
  }

  // Override user_conversion with dt_conversion.
  if (user_conversion != dt_conversion) {
    if (!user_conversion.empty() &&
        user_conversion.back() != dt_conversion.back()) {
      // Preserves user_conversion when user and dt conversions belong to the
      // same group, e.g., when printing unsigned decimal numbers in hexadecimal
      // or octal format, or floating point numbers in exponential notation.
      if ((signed_group.find(user_conversion.back()) != std::string::npos &&
           signed_group.find(dt_conversion.back()) != std::string::npos) ||
          (unsigned_group.find(user_conversion.back()) != std::string::npos &&
           unsigned_group.find(dt_conversion.back()) != std::string::npos) ||
          (float_group.find(user_conversion.back()) != std::string::npos &&
           float_group.find(dt_conversion.back()) != std::string::npos)) {
        dt_conversion.back() = user_conversion.back();
      } else {
        TI_WARN("The printf conversion specifier '{}' is overritten by '{}'",
                user_conversion, dt_conversion);
      }
    }
    user_conversion = dt_conversion;
  }

  std::string res = "%" + user_flags + user_width + user_precision +
                    user_length + user_conversion;
  TI_TRACE("Merge %{} and {} into {}.", user, from_data_type, res);
  return res;
}

}  // namespace taichi::lang
