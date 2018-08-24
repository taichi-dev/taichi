/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <vector>
#include <string>
#include "interface.h"

TC_NAMESPACE_BEGIN

class Task : public Unit {
 public:
  virtual std::string run(const std::vector<std::string> &parameters) {
    assert_info(parameters.size() == 0, "No parameters supported.");
    return this->run();
  }
  virtual std::string run() {
    return this->run(std::vector<std::string>());
  }
};

TC_INTERFACE(Task)

inline std::string task_invoke(
    const std::function<std::string(const std::vector<std::string>)> &func,
    const std::vector<std::string> &params) {
  return func(params);
}

inline std::string task_invoke(
    const std::function<void(const std::vector<std::string>)> &func,
    const std::vector<std::string> &params) {
  func(params);
  return "";
}

inline std::string task_invoke(const std::function<void()> &func,
                               const std::vector<std::string> &params) {
  func();
  return "";
}

inline std::string task_invoke(const std::function<std::string()> &func,
                               const std::vector<std::string> &params) {
  return func();
}

#define TC_REGISTER_TASK(task)                                             \
  class Task_##task : public taichi::Task {                                \
    std::string run(const std::vector<std::string> &parameters) override { \
      return task_invoke(task, parameters);                                     \
    }                                                                      \
  };                                                                       \
  TC_IMPLEMENTATION(Task, Task_##task, #task)

TC_NAMESPACE_END
