#pragma once

#include "taichi/rhi/device.h"
#include "taichi/util/lang_util.h"

namespace taichi::lang {

class Program;
class IRNode;
class FrontendContext;

class TI_DLL_EXPORT CallableBase {
 public:
  struct Parameter {
    bool is_array{
        false};  // This is true for both ndarray and external array args.
    std::size_t total_dim{0};  // total dim of array
    BufferFormat format{BufferFormat::unknown};

    TI_IO_DEF(is_array, total_dim, format, dt_);

    bool operator==(const Parameter &o) const {
      return is_array == o.is_array && total_dim == o.total_dim &&
             format == o.format && dt_ == o.dt_;
    }

    /* [arguments with TensorType]

    Taichi used to represent TensorType with the combination of "PrimitiveType"
    & "element_shape" and there are a bunch of interfaces designed like this (it
    allows creating TensorType by passing in PrimitiveType + element_shape)

    Here we removed the "element_shape" member in the underlying objects (class
    Arg, class ExternalTensorExpression, ...), and forced them to use TensorType
    in their "dtype" member.

    However we kept the interfaces unchanged temporarily, so as to minimize
    possible regressions.
    */
    explicit Parameter(const DataType &dt = PrimitiveType::unknown,
                       bool is_array = false,
                       std::size_t size_unused = 0,
                       int total_dim = 0,
                       std::vector<int> element_shape = {},
                       BufferFormat format = BufferFormat::unknown) {
      if (dt->is<PrimitiveType>() && element_shape.size() > 0) {
        this->dt_ =
            taichi::lang::TypeFactory::get_instance().create_tensor_type(
                element_shape, dt);
      } else {
        this->dt_ = dt;
      }

      this->is_array = is_array;
      this->total_dim = total_dim;
      this->format = format;
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

    TI_IO_DEF(dt);

    explicit Ret(const DataType &dt = PrimitiveType::unknown) : dt(dt) {
    }
  };

  std::vector<Parameter> parameter_list;
  std::vector<Ret> rets;

  const StructType *ret_type = nullptr;
  size_t ret_size{0};

  const StructType *args_type = nullptr;
  size_t args_size{0};

  Arch arch;
  std::string name;
};

class TI_DLL_EXPORT Callable : public CallableBase {
 public:
  Program *program{nullptr};
  std::unique_ptr<IRNode> ir{nullptr};
  std::unique_ptr<FrontendContext> context{nullptr};

  Callable();
  virtual ~Callable();

  int insert_scalar_param(const DataType &dt);

  int insert_arr_param(const DataType &dt,
                       int total_dim,
                       std::vector<int> element_shape);
  int insert_texture_param(int total_dim);
  int insert_rw_texture_param(int total_dim, BufferFormat format);

  int insert_ret(const DataType &dt);

  void finalize_rets();

  void finalize_params();

  [[nodiscard]] virtual std::string get_name() const = 0;
};

}  // namespace taichi::lang
