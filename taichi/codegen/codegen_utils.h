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
  std::regex_match(from_user.value(), user_match, user_re);
  std::string user_flags = user_match[1];
  std::string user_width = user_match[2];
  std::string user_precision = user_match[3];
  std::string user_length = user_match[4];
  std::string user_conversion = user_match[5];

  // Assert that there are no dangeous specifiers.
  if (user_width == "*" || user_precision == ".*" || user_conversion == "n") {
    TI_NOT_IMPLEMENTED
  }

  const std::regex dt_re = std::regex(
      "%"
      "(\\.(?:\\d+))?"
      "([hljztL]|hh|ll)?"
      "([csdioxXufFeEaAgGnp])?");
  std::smatch dt_match;
  std::regex_match(from_data_type, dt_match, dt_re);
  std::string dt_precision = dt_match[1];
  std::string dt_length = dt_match[2];
  std::string dt_conversion = dt_match[3];

  // Vulkan doesn't support length, flags, or width specifier.
  // https://vulkan.lunarg.com/doc/view/1.2.162.1/linux/debug_printf.html
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
    if (!user_length.empty()) {
      TI_WARN(
          "The printf length modifier '{}' is not supported in Vulkan, "
          "and will be discarded.",
          user_length);
      user_length.clear();
    }
  }
  // CUDA supports all of them.
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#format-specifiers
  else if (arch == Arch::cuda) {
    // DO NOTHING
  }

  // Respect user_precision if and only if dt_conversion is float and
  // dt_precision is empty.
  TI_ASSERT(!dt_conversion.empty());
  if (!(dt_conversion.back() == 'f' && dt_precision.empty()) &&
      user_precision != dt_precision) {
    TI_WARN("The printf precision specifier '{}' is overritten by '{}'",
            user_precision, dt_precision);
    user_precision = dt_precision;
  }

  // Discard user_length and give warning if it doesn't match with dt_length.
  if (!user_length.empty() && user_length != dt_length) {
    TI_WARN("The printf length specifier '{}' is overritten by '{}'",
            user_length, dt_length);
    user_length = dt_length;
  }

  // Override user_conversion with dt_conversion.
  if (!user_conversion.empty() &&
      user_conversion.back() != dt_conversion.back()) {
    // Preserves user_conversion when user and dt conversions belong to the same
    // group, e.g., when printing unsigned decimal numbers in hexadecimal
    // or octal format, or floating point numbers in exponential notation.
    constexpr std::string_view signed_group = "di";
    constexpr std::string_view unsigned_group = "oxXu";
    constexpr std::string_view float_group = "fFeEaAgG";

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
  TI_ASSERT(!dt_conversion.empty());
  user_conversion = dt_conversion;

  std::string res = "%" + user_flags + user_width + user_length +
                    user_precision + user_conversion;
  TI_TRACE("Merge %{} and {} into {}.", from_user.value(), from_data_type, res);
  return res;
}

}  // namespace taichi::lang
