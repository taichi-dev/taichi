#pragma once

#include "taichi/lang_util.h"

namespace taichi {
namespace lang {

class Program;
class IRNode;

class Callable {
 public:
  Program *program;
  std::unique_ptr<IRNode> ir;

  struct Arg {
    DataType dt;
    bool is_external_array;
    std::size_t size;

    explicit Arg(DataType dt = PrimitiveType::unknown,
                 bool is_external_array = false,
                 std::size_t size = 0)
        : dt(dt), is_external_array(is_external_array), size(size) {
    }
  };

  struct Ret {
    DataType dt;

    explicit Ret(DataType dt = PrimitiveType::unknown) : dt(dt) {
    }
  };

  std::vector<Arg> args;
  std::vector<Ret> rets;

  virtual ~Callable() = default;

  int insert_arg(DataType dt, bool is_external_array);

  int insert_ret(DataType dt);
};

}  // namespace lang
}  // namespace taichi
