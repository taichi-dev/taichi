#pragma once

#include "taichi/util/lang_util.h"

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
    bool is_array{
        false};  // This is true for both ndarray and external array args.
    std::size_t total_dim{0};  // total dim of array

    explicit Arg(const DataType &dt = PrimitiveType::unknown,
                 bool is_array = false,
                 std::size_t size_unused = 0,
                 int total_dim = 0,
                 std::vector<int> element_shape = {}) {
      if (dt->is<PrimitiveType>() && element_shape.size() > 0) {
        this->dt_ =
            taichi::lang::TypeFactory::get_instance().create_tensor_type(
                element_shape, dt);

      } else {
        this->dt_ = dt;
      }

      this->is_array = is_array;
      this->total_dim = total_dim;
    }

    std::vector<int> get_element_shape() const {
      return dt_.get_shape();
    }

    DataType get_element_type() const {
      return dt_.get_element_type();
    }

    int get_element_size() const {
      return data_type_size(dt_);
    }

    DataType get_dtype() const {
      return dt_;
    }

   private:
    DataType dt_;
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

  int insert_scalar_arg(const DataType &dt);

  int insert_arr_arg(const DataType &dt,
                     int total_dim,
                     std::vector<int> element_shape);
  int insert_texture_arg(const DataType &dt);

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
