#pragma once

#include "taichi/program/callable.h"
#include "taichi/program/function_key.h"

namespace taichi {
namespace lang {

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

  std::optional<std::string> &try_get_ast_serialization_data() {
    return ast_serialization_data_;
  }

 private:
  std::optional<std::string> ast_serialization_data_;  // For generating AST-Key
};

}  // namespace lang
}  // namespace taichi
