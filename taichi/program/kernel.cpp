#include "taichi/program/kernel.h"

#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/codegen/codegen.h"
#include "taichi/common/logging.h"
#include "taichi/common/task.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/util/action_recorder.h"

#ifdef TI_WITH_LLVM
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif

namespace taichi::lang {

class Function;

Kernel::Kernel(Program &program,
               const std::function<void()> &func,
               const std::string &primal_name,
               AutodiffMode autodiff_mode) {
  this->init(program, func, primal_name, autodiff_mode);
}

Kernel::Kernel(Program &program,
               const std::function<void(Kernel *)> &func,
               const std::string &primal_name,
               AutodiffMode autodiff_mode) {
  // due to #6362, we cannot write [func, this] { return func(this); }
  this->init(
      program, [&] { return func(this); }, primal_name, autodiff_mode);
}

Kernel::Kernel(Program &program,
               std::unique_ptr<IRNode> &&ir,
               const std::string &primal_name,
               AutodiffMode autodiff_mode)
    : autodiff_mode(autodiff_mode), lowered_(false) {
  this->ir = std::move(ir);
  this->program = &program;
  is_accessor = false;
  is_evaluator = false;
  compiled_ = nullptr;
  ir_is_ast_ = false;  // CHI IR

  if (autodiff_mode == AutodiffMode::kNone) {
    name = primal_name;
  } else if (autodiff_mode == AutodiffMode::kForward) {
    name = primal_name + "_forward_grad";
  } else if (autodiff_mode == AutodiffMode::kReverse) {
    name = primal_name + "_reverse_grad";
  }
}

void Kernel::compile(const CompileConfig &compile_config) {
  compiled_ = program->compile(compile_config, *this);
}

void Kernel::operator()(const CompileConfig &compile_config,
                        LaunchContextBuilder &ctx_builder) {
  if (!compiled_) {
    compile(compile_config);
  }

  auto &context = ctx_builder.get_context();
  program->prepare_runtime_context(&context);
  compiled_(context);

  const auto arch = compile_config.arch;
  if (compile_config.debug &&
      (arch_is_cpu(arch) || arch == Arch::cuda || arch == Arch::amdgpu)) {
    program->check_runtime_error();
  }
}

LaunchContextBuilder Kernel::make_launch_context() {
  return LaunchContextBuilder(this);
}

template <typename T>
T Kernel::fetch_ret(DataType dt, int i) {
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return (T)program->fetch_result<float32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return (T)program->fetch_result<float64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return (T)program->fetch_result<int32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return (T)program->fetch_result<int64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return (T)program->fetch_result<int8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return (T)program->fetch_result<int16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return (T)program->fetch_result<uint8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return (T)program->fetch_result<uint16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return (T)program->fetch_result<uint32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return (T)program->fetch_result<uint64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
    // use f32 to interact with python
    return (T)program->fetch_result<float32>(i);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

float64 Kernel::get_ret_float(int i) {
  auto dt = rets[i].dt->get_compute_type();
  return fetch_ret<float64>(dt, i);
}

int64 Kernel::get_ret_int(int i) {
  auto dt = rets[i].dt->get_compute_type();
  return fetch_ret<int64>(dt, i);
}

uint64 Kernel::get_ret_uint(int i) {
  auto dt = rets[i].dt->get_compute_type();
  return fetch_ret<uint64>(dt, i);
}

std::vector<int64> Kernel::get_ret_int_tensor(int i) {
  DataType dt = rets[i].dt->as<TensorType>()->get_element_type();
  int size = rets[i].dt->as<TensorType>()->get_num_elements();
  std::vector<int64> res;
  for (int j = 0; j < size; j++) {
    res.emplace_back(fetch_ret<int64>(dt, j));
  }
  return res;
}

std::vector<uint64> Kernel::get_ret_uint_tensor(int i) {
  DataType dt = rets[i].dt->as<TensorType>()->get_element_type();
  int size = rets[i].dt->as<TensorType>()->get_num_elements();
  std::vector<uint64> res;
  for (int j = 0; j < size; j++) {
    res.emplace_back(fetch_ret<uint64>(dt, j));
  }
  return res;
}

std::vector<float64> Kernel::get_ret_float_tensor(int i) {
  DataType dt = rets[i].dt->as<TensorType>()->get_element_type();
  int size = rets[i].dt->as<TensorType>()->get_num_elements();
  std::vector<float64> res;
  for (int j = 0; j < size; j++) {
    res.emplace_back(fetch_ret<float64>(dt, j));
  }
  return res;
}

std::string Kernel::get_name() const {
  return name;
}

void Kernel::init(Program &program,
                  const std::function<void()> &func,
                  const std::string &primal_name,
                  AutodiffMode autodiff_mode) {
  this->autodiff_mode = autodiff_mode;
  this->lowered_ = false;
  this->program = &program;

  is_accessor = false;
  is_evaluator = false;
  compiled_ = nullptr;
  context = std::make_unique<FrontendContext>(program.compile_config().arch);
  ir = context->get_root();
  ir_is_ast_ = true;
  arch = program.compile_config().arch;

  if (autodiff_mode == AutodiffMode::kNone) {
    name = primal_name;
  } else if (autodiff_mode == AutodiffMode::kCheckAutodiffValid) {
    name = primal_name + "_validate_grad";
  } else if (autodiff_mode == AutodiffMode::kForward) {
    name = primal_name + "_forward_grad";
  } else if (autodiff_mode == AutodiffMode::kReverse) {
    name = primal_name + "_reverse_grad";
  }

  func();
}
}  // namespace taichi::lang
