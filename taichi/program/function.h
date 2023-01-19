#pragma once

#include "taichi/program/callable.h"
#include "taichi/program/function_key.h"

namespace taichi::lang {

class Program;

class Function : public Callable {
 public:
  enum class IRType { None, AST, InitialIR, OptimizedIR };

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

  void set_ir_type(IRType type) {
    ir_type_ = type;
  }

  IRType ir_type() const {
    return ir_type_;
  }

 private:
  IRType ir_type_{IRType::None};
  std::optional<std::string> ast_serialization_data_;  // For generating AST-Key
};

}  // namespace taichi::lang
