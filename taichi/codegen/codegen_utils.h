#pragma once
#include "taichi/program/program.h"
#include "taichi/util/lang_util.h"

namespace taichi::lang {

inline bool codegen_vector_type(const CompileConfig &config) {
  return !(config.real_matrix_scalarize || config.force_scalarize_matrix);
}

// Parses a C-style printf format string specifier into its constituent parts.
inline std::array<std::string, 5> parse_printf_specifier(std::string spec) {
  // The format string specifiers are of the form:
  // %[<flags>]+[<width>][.<precision>][<length>]<conversion>
  // Where conversion is required in C/C++, others are optional.
  // See https://en.cppreference.com/w/cpp/io/c/fprintf
  // Note that in taichi we support omitting the conversion, and the leading '%'
  // is ignored.
  const std::regex re = std::regex(
      "%?"
      "([-+ #0]+)?"
      "(\\d+|\\*)?"
      "(\\.(?:\\d+|\\*))?"
      "([hljztL]|hh|ll)?"
      "([csdioxXufFeEaAgGnp])?");
  std::smatch match;
  bool matched = std::regex_match(spec, match, re);
  if (matched == false) {
    TI_ERROR("{} is not a valid printf specifier.", spec)
  }
  std::string flags = match[1];
  std::string width = match[2];
  std::string precision = match[3];
  std::string length = match[4];
  std::string conversion = match[5];

  return {flags, width, precision, length, conversion};
}

// Merges format specifiers with respect to user specified ones and taichi
// inferred ones.
// For example, when printing an f64:
//   1. user may input:
//     a. '.2f' (the canonical way to control precesion),
//     b. 'e' (use decimal exponent notation),
//     c. '-+10lf' (left align, show sign explicitly, minimum field width is 10
//        characters, print an double),
//     d. '.2' (only specify precision, use default conversion),
//     e. nullopt_t (default),
//   as the specifier.
//   2. taichi will infer the specifier as '%.12f'.
// We should merge them to:
//   a. '%.2f',
//   b. '%.12e',
//   c. '%-+10.12f' ('lf' means the same thing as 'f'),
//   d. '%.2f',
//   e. '%.12f',
// accordingly.
inline std::string merge_printf_specifier(
    std::optional<std::string> const &from_user,
    std::string const &from_data_type) {
  if (!from_user.has_value()) {
    return from_data_type;
  }
  std::string const &user = from_user.value();

  auto &&[user_flags, user_width, user_precision, user_length,
          user_conversion] = parse_printf_specifier(user);
  if (user_width == "*" || user_precision == ".*" || user_conversion == "n") {
    TI_ERROR("The {} printf specifier is not supported", user)
  }

  auto &&[_, __, dt_precision, dt_length, dt_conversion] =
      parse_printf_specifier(from_data_type);

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

  // Constant for convensions in group.
  constexpr std::string_view signed_group = "di";
  constexpr std::string_view unsigned_group = "oxXu";
  constexpr std::string_view float_group = "fFeEaAgG";

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
