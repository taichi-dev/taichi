#pragma once
#include "taichi/program/program.h"

namespace taichi::lang {

inline bool codegen_vector_type(const CompileConfig &config) {
  return !config.real_matrix_scalarize;
}

inline std::string merge_format_data_type(
    std::optional<std::string> const &from_format,
    std::string const &from_data_type) {
  if (!from_format.has_value()) {
    return from_data_type;
  }

  constexpr std::string_view data_type_specifier = "hdluf";

  std::string const &format = from_format.value();
  size_t fmt_dt_pivot = format.find_first_of(data_type_specifier);
  std::string fmt = format.substr(0, fmt_dt_pivot);
  std::string fmt_dt = format.substr(fmt_dt_pivot);

  std::string dt = from_data_type.substr(1);

  if (!fmt_dt.empty() && fmt_dt.back() != dt.back()) {
    TI_WARN("data type doesn't match between %{} and %{}", fmt_dt, dt);
  }

  if (dt.find('.') != dt.npos && fmt.find('.') != fmt.npos) {
    // if dt gives precesion, then discard precesion from fmt
    fmt = fmt.substr(0, fmt.find('.'));
  }

  std::string res = "%" + fmt + dt;
  return res;
}

}  // namespace taichi::lang
