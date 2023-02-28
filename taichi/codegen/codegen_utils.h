#pragma once
#include "taichi/program/program.h"

namespace taichi::lang {

inline bool codegen_vector_type(const CompileConfig &config) {
  return !config.real_matrix_scalarize;
}

inline std::string merge_printf_specifier(
    std::optional<std::string> const &from_format,
    std::string const &from_data_type) {
  if (!from_format.has_value()) {
    return from_data_type;
  }

  constexpr std::string_view printf_specifier_and_length =
      "diuoxXfFeFgGaAcspn%hljztL";

  std::string const &format = from_format.value();
  size_t fmt_dt_pivot = format.find_first_of(printf_specifier_and_length);
  std::string fmt = format.substr(0, fmt_dt_pivot);
  std::string fmt_dt =
      fmt_dt_pivot == format.npos ? "" : format.substr(fmt_dt_pivot);

  std::string dt = from_data_type.substr(1);

  // The user-defined data type is overridden by the type specifier inferred by
  // taichi.
  if (!fmt_dt.empty() && fmt_dt.back() != dt.back()) {
    // Respect fmt_dt in some corner cases, e.g., when printing decimal
    // numbers using hex or oct format.
    if ((fmt_dt.back() == 'o' || fmt_dt.back() == 'x') && (dt.back() == 'd')) {
      dt.back() = fmt_dt.back();
    } else {
      TI_WARN("printf specifier %{} overridden by %{}", fmt_dt, dt);
    }
  }

  // If dt gives the precision, then discard the precision from fmt.
  if (dt.find('.') != dt.npos && fmt.find('.') != fmt.npos) {
    fmt = fmt.substr(0, fmt.find('.'));
  }

  std::string res = "%" + fmt + dt;
  return res;
}

}  // namespace taichi::lang
