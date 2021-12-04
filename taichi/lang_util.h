// Definitions of utility functions and enums
#pragma once

#include "taichi/util/io.h"
#include "taichi/common/core.h"
#include "taichi/system/profiler.h"
#include "taichi/common/exceptions.h"
#include "taichi/ir/stmt_op_types.h"
#include "taichi/ir/type.h"
#include "taichi/ir/type_utils.h"
#include "taichi/ir/type_factory.h"

TLANG_NAMESPACE_BEGIN

real get_cpu_frequency();

extern real default_measurement_time;

real measure_cpe(std::function<void()> target,
                 int64 elements_per_call,
                 real time_second = default_measurement_time);

struct RuntimeContext;

using FunctionType = std::function<void(RuntimeContext &)>;

inline std::string make_list(const std::vector<std::string> &data,
                             std::string bracket = "") {
  std::string ret = bracket;
  for (int i = 0; i < (int)data.size(); i++) {
    ret += data[i];
    if (i + 1 < (int)data.size()) {
      ret += ", ";
    }
  }
  if (bracket == "<") {
    ret += ">";
  } else if (bracket == "{") {
    ret += "}";
  } else if (bracket == "[") {
    ret += "]";
  } else if (bracket == "(") {
    ret += ")";
  } else if (bracket != "") {
    TI_P(bracket);
    TI_NOT_IMPLEMENTED
  }
  return ret;
}

template <typename T>
std::string make_list(const std::vector<T> &data,
                      std::function<std::string(const T &t)> func,
                      std::string bracket = "") {
  std::vector<std::string> ret(data.size());
  for (int i = 0; i < (int)data.size(); i++) {
    ret[i] = func(data[i]);
  }
  return make_list(ret, bracket);
}

extern std::string compiled_lib_dir;
extern std::string runtime_tmp_dir;
std::string runtime_lib_dir();

bool command_exist(const std::string &command);

TLANG_NAMESPACE_END

TI_NAMESPACE_BEGIN
void initialize_benchmark();

template <typename T, typename... Args, typename FP = T (*)(Args...)>
FP function_pointer_helper(std::function<T(Args...)>) {
  return nullptr;
}

template <typename T, typename... Args, typename FP = T (*)(Args...)>
FP function_pointer_helper(T (*)(Args...)) {
  return nullptr;
}

template <typename T>
using function_pointer_type =
    decltype(function_pointer_helper(std::declval<T>()));

TI_NAMESPACE_END
