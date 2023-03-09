#pragma once
#include "taichi/program/program.h"
#include "taichi/util/lang_util.h"

namespace taichi::lang {

inline bool codegen_vector_type(const CompileConfig &config) {
  return !config.real_matrix_scalarize;
}

inline std::string merge_printf_specifier(
    std::optional<std::string> const &from_user,
    std::string const &from_data_type,
    Arch arch) {
  // TODO: implement specifier merging
  // C-style printf format string specifiers:
  // %[<flags>]+[<width>][.<precision>][<length>]<conversion>
  // Where conversion is required, others are optional.
  // See https://en.cppreference.com/w/cpp/io/c/fprintf

  // For example, when printing an f64:
  // 1. user may input:
  //   a. '.2f' (the canonical way to control precesion),
  //   b. 'e' (use decimal exponent notation),
  //   c. '-+10lf' (left align, show sign explicitly, minimum field width is 10
  //      characters, print an double),
  //   d. nullopt_t (default),
  // as the specifier.
  // 2. taichi will infer the specifier as '%.12f'.

  // We should merge them to:
  //   a. '%.2f',
  //   b. '%.12e',
  //   c. '%-+10.12f' ('lf' means the same thing as 'f'),
  //   d. '%.12f',
  // accordingly.

  return from_data_type;
}

}  // namespace taichi::lang
