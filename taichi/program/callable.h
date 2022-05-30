#pragma once

#include "taichi/lang_util.h"

namespace taichi {
namespace lang {

class Program;
class IRNode;
class FrontendContext;

class TI_DLL_EXPORT Callable {
 public:
  Program *program{nullptr};
  std::unique_ptr<IRNode> ir{nullptr};
  std::unique_ptr<FrontendContext> context{nullptr};

  struct Arg {
    DataType dt;
    bool is_array{
        false};  // This is true for both ndarray and external array args.
    std::size_t total_dim{0};             // total dim of array
    std::vector<int> element_shape = {};  // shape of each element

    explicit Arg(const DataType &dt = PrimitiveType::unknown,
                 bool is_array = false,
                 std::size_t size_unused = 0,
                 int total_dim = 0,
                 std::vector<int> element_shape = {})
        : dt(dt),
          is_array(is_array),
          total_dim(total_dim),
          element_shape(std::move(element_shape)) {
    }
  };

  struct Ret {
    DataType dt;

    explicit Ret(const DataType &dt = PrimitiveType::unknown) : dt(dt) {
    }
  };

  std::vector<Arg> args;
  std::vector<Ret> rets;

  Callable();
  virtual ~Callable();

  int insert_arg(const DataType &dt, bool is_array);

  int insert_arr_arg(const DataType &dt,
                     int total_dim,
                     std::vector<int> element_shape);

  int insert_ret(const DataType &dt);

  [[nodiscard]] virtual std::string get_name() const = 0;

  class CurrentCallableGuard {
    Callable *old_callable_;
    Program *program_;

   public:
    CurrentCallableGuard(Program *program, Callable *callable);

    ~CurrentCallableGuard();
  };
};

}  // namespace lang
}  // namespace taichi
