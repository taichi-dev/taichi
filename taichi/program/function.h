#pragma once

#include "taichi/program/callable.h"
#include "taichi/program/function_key.h"

namespace taichi::lang {

class Program;

class Function : public Callable {
 public:
  FunctionKey func_key;

  Function(Program *program, const FunctionKey &func_key);

  // Set the function body to a frontend Python function which generates the C++
  // AST.
  void set_function_body(const std::function<void()> &func);

  // Set the function body to a CHI IR.
  void set_function_body(std::unique_ptr<IRNode> func_body);

  [[nodiscard]] std::string get_name() const override;

  const std::optional<std::string> &try_get_ast_serialization_data() const {
    return ast_serialization_data_;
  }

  void set_ast_serialization_data(std::string ast_data) {
    ast_serialization_data_ = std::move(ast_data);
  }

  bool lowered() const {
    return lowered_;
  }

  void set_lowered(bool lowered) {
    lowered_ = lowered;
  }

  bool ir_start_from_ast() const {
    return ir_start_from_ast_;
  }

 private:
  bool ir_start_from_ast_{false};  // Refactor2023:FIXME: Remove it
  bool lowered_{false};
  std::optional<std::string> ast_serialization_data_;  // For generating AST-Key
};

}  // namespace taichi::lang
