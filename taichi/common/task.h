/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <vector>
#include <string>

#include "taichi/common/interface.h"

TI_NAMESPACE_BEGIN

class Task : public Unit {
 public:
  virtual std::string run(const std::vector<std::string> &parameters) {
    TI_ASSERT_INFO(parameters.size() == 0, "No parameters supported.");
    return this->run();
  }

  virtual std::string run() {
    return this->run(std::vector<std::string>());
  }

  ~Task() {
  }
};

TI_INTERFACE(Task)

template <typename T>
inline std::enable_if_t<
    std::is_same<std::result_of_t<T(const std::vector<std::string> &)>,
                 void>::value ||
        std::is_same<std::result_of_t<T(const std::vector<std::string> &)>,
                     const char *>::value,
    std::string>
task_invoke(const T &func, const std::vector<std::string> &params) {
  func(params);
  return "";
}

template <typename T>
inline std::enable_if_t<
    std::is_same<std::result_of_t<T(const std::vector<std::string> &)>,
                 std::string>::value,
    std::string>
task_invoke(const T &func, const std::vector<std::string> &params) {
  return func(params);
}

template <typename T>
inline std::enable_if_t<std::is_same<std::result_of_t<T()>, void>::value,
                        std::string>
task_invoke(const T &func, const std::vector<std::string> &params) {
  func();
  return "";
}

template <typename T>
inline std::enable_if_t<
    std::is_same<std::result_of_t<T()>, std::string>::value ||
        std::is_same<std::result_of_t<T()>, const char *>::value,
    std::string>
task_invoke(const T &func, const std::vector<std::string> &params) {
  return func();
}

#define TI_REGISTER_TASK(task)                                             \
  class Task_##task : public taichi::Task {                                \
    std::string run(const std::vector<std::string> &parameters) override { \
      return taichi::task_invoke<decltype(task)>(task, parameters);        \
    }                                                                      \
  };                                                                       \
  TI_IMPLEMENTATION(Task, Task_##task, #task)

TI_NAMESPACE_END
