#pragma once
#include "taichi/inc/constants.h"
#include "taichi/rhi/device.h"
#include "taichi/util/lang_util.h"

#include <stack>

namespace taichi::lang {

class Program;
class IRNode;
class FrontendContext;

class TI_DLL_EXPORT CallableBase {
 public:
  struct Parameter {
    std::string name;
    bool is_array{
        false};  // This is true for both ndarray and external array args.
    bool is_argpack{false};
    std::size_t total_dim{0};  // total dim of array
    BufferFormat format{BufferFormat::unknown};
    bool needs_grad{false};  // TODO: reorder for better alignment
    std::vector<int> element_shape{};
    ParameterType ptype{ParameterType::kUnknown};
    TI_IO_DEF(is_array,
              is_argpack,
              total_dim,
              format,
              dt_,
              needs_grad,
              element_shape,
              ptype);

    bool operator==(const Parameter &o) const {
      return is_array == o.is_array && total_dim == o.total_dim &&
             format == o.format && dt_ == o.dt_ && needs_grad == o.needs_grad &&
             element_shape == o.element_shape && ptype == o.ptype &&
             is_argpack == o.is_argpack;
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
                       bool is_argpack = false,
                       std::size_t size_unused = 0,
                       int total_dim = 0,
                       std::vector<int> element_shape = {},
                       BufferFormat format = BufferFormat::unknown,
                       bool needs_grad = false) {
      // TODO: Currently dt is only PrimitiveType or StructType for
      // ndarray/texture/matrix
      //       We should always keep it either PrimitiveType or TensorType. In
      //       other words, `get_type_for_kernel_args` which we currently do in
      //       Python should be delayed until finalize_params.
      if (dt->is<PrimitiveType>() && element_shape.size() > 0) {
        this->dt_ =
            taichi::lang::TypeFactory::get_instance().create_tensor_type(
                element_shape, dt);
      } else {
        this->dt_ = dt;
      }
      this->element_shape = element_shape;
      this->is_array = is_array;
      this->is_argpack = is_argpack;
      this->total_dim = total_dim;
      this->format = format;
      this->needs_grad = needs_grad;
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
  // Note: `nested_parameters` stores not only nested parameters, but also
  // those parameters in `parameter_list`.
  std::unordered_map<std::vector<int>,
                     Parameter,
                     hashing::Hasher<std::vector<int>>>
      nested_parameters;
  std::unordered_map<std::vector<int>,
                     const StructType *,
                     hashing::Hasher<std::vector<int>>>
      argpack_types;
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
  AutodiffMode autodiff_mode{AutodiffMode::kNone};

  Callable();
  virtual ~Callable();

  std::vector<int> insert_scalar_param(const DataType &dt,
                                       const std::string &name = "");
  std::vector<int> insert_arr_param(const DataType &dt,
                                    int total_dim,
                                    std::vector<int> element_shape,
                                    const std::string &name = "");
  std::vector<int> insert_ndarray_param(const DataType &dt,
                                        int ndim,
                                        const std::string &name = "",
                                        bool needs_grad = false);
  std::vector<int> insert_texture_param(int total_dim,
                                        const std::string &name = "");
  std::vector<int> insert_pointer_param(const DataType &dt,
                                        const std::string &name = "");
  std::vector<int> insert_rw_texture_param(int total_dim,
                                           BufferFormat format,
                                           const std::string &name = "");

  std::vector<int> insert_argpack_param_and_push(const std::string &name = "");

  void pop_argpack_stack();

  int insert_ret(const DataType &dt);

  void finalize_rets();

  void finalize_params();

  [[nodiscard]] virtual std::string get_name() const = 0;

 private:
  std::vector<int> add_parameter(const Parameter &param);
  // Note: These stacks are used for inserting params inside argpacks. When
  // we call finalize_params(), all of them are required to be empty then.
  std::stack<std::vector<Parameter>> temp_argpack_stack_;
  std::vector<int> temp_indices_stack_;
  std::stack<std::string> temp_argpack_name_stack_;
};

}  // namespace taichi::lang
