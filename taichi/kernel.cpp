#include <taichi/common/task.h>
#include "kernel.h"
#include "program.h"
#if defined(CUDA_FOUND)
#include <cuda_runtime.h>
#include "cuda_utils.h"
#endif

TLANG_NAMESPACE_BEGIN

Kernel::Kernel(Program &program,
               std::function<void()> func,
               std::string name,
               bool grad)
    : program(program), name(name), grad(grad) {
  program.initialize_device_llvm_context();
  is_accessor = false;
  is_reduction = false;
  compiled = nullptr;
  benchmarking = false;
  taichi::Tlang::context = std::make_unique<FrontendContext>();
  ir_holder = taichi::Tlang::context->get_root();
  ir = ir_holder.get();

  program.current_kernel = this;
  program.start_function_definition(this);
  func();
  program.end_function_definition();
  program.current_kernel = nullptr;

  arch = program.config.arch;

  if (!program.config.lazy_compilation)
    compile();
}

void Kernel::compile() {
  program.current_kernel = this;
  compiled = program.compile(*this);
  program.current_kernel = nullptr;
}

void Kernel::operator()() {
  if (!compiled)
    compile();
  if (arch == Arch::cuda) {
    std::vector<void *> host_buffers(args.size());
    std::vector<void *> device_buffers(args.size());
#if defined(CUDA_FOUND)
    // copy data to GRAM
    bool has_buffer = false;
    for (int i = 0; i < (int)args.size(); i++) {
      if (args[i].is_nparray) {
        has_buffer = true;
        check_cuda_errors(cudaMalloc(&device_buffers[i], args[i].size));
        // replace host buffer with device buffer
        host_buffers[i] = program.context.get_arg<void *>(i);
        set_arg_nparray(i, (uint64)device_buffers[i], args[i].size);
        check_cuda_errors(cudaMemcpy(device_buffers[i], host_buffers[i],
                                     args[i].size, cudaMemcpyHostToDevice));
      }
    }
    if (has_buffer)
      check_cuda_errors(cudaDeviceSynchronize());
    auto c = program.get_context();
    compiled(c);
    if (has_buffer)
      check_cuda_errors(cudaDeviceSynchronize());
    for (int i = 0; i < (int)args.size(); i++) {
      if (args[i].is_nparray) {
        check_cuda_errors(cudaMemcpy(host_buffers[i], device_buffers[i],
                                     args[i].size, cudaMemcpyDeviceToHost));
        check_cuda_errors(cudaFree(device_buffers[i]));
      }
    }
#else
    TC_ERROR("No CUDA");
#endif
  } else {
    auto &c = program.get_context();
    compiled(c);
  }
  program.sync = false;
}

void Kernel::set_arg_float(int i, float64 d) {
  TC_ASSERT_INFO(args[i].is_nparray == false,
                 "Setting scalar value to numpy array argument is not allowed");
  auto dt = args[i].dt;
  if (dt == DataType::f32) {
    program.context.set_arg(i, (float32)d);
  } else if (dt == DataType::f64) {
    program.context.set_arg(i, (float64)d);
  } else if (dt == DataType::i32) {
    program.context.set_arg(i, (int32)d);
  } else if (dt == DataType::i64) {
    program.context.set_arg(i, (int64)d);
  } else if (dt == DataType::i16) {
    program.context.set_arg(i, (int16)d);
  } else if (dt == DataType::u16) {
    program.context.set_arg(i, (uint16)d);
  } else if (dt == DataType::u32) {
    program.context.set_arg(i, (uint32)d);
  } else if (dt == DataType::u64) {
    program.context.set_arg(i, (uint64)d);
  } else {
    TC_NOT_IMPLEMENTED
  }
}

void Kernel::set_extra_arg_int(int i, int j, int32 d) {
  program.context.extra_args[i][j] = d;
}

void Kernel::set_arg_int(int i, int64 d) {
  TC_ASSERT_INFO(args[i].is_nparray == false,
                 "Setting scalar value to numpy array argument is not allowed");
  auto dt = args[i].dt;
  if (dt == DataType::i32) {
    program.context.set_arg(i, (int32)d);
  } else if (dt == DataType::i64) {
    program.context.set_arg(i, (int64)d);
  } else if (dt == DataType::i16) {
    program.context.set_arg(i, (int16)d);
  } else if (dt == DataType::u16) {
    program.context.set_arg(i, (uint16)d);
  } else if (dt == DataType::u32) {
    program.context.set_arg(i, (uint32)d);
  } else if (dt == DataType::u64) {
    program.context.set_arg(i, (uint64)d);
  } else if (dt == DataType::f32) {
    program.context.set_arg(i, (float32)d);
  } else if (dt == DataType::f64) {
    program.context.set_arg(i, (float64)d);
  } else {
    TC_NOT_IMPLEMENTED
  }
}

void Kernel::mark_arg_return_value(int i, bool is_return) {
  args[i].is_return_value = is_return;
}

void Kernel::set_arg_nparray(int i, uint64 d, uint64 size) {
  TC_ASSERT_INFO(args[i].is_nparray,
                 "Setting numpy array to scalar argument is not allowed");
  args[i].size = size;
  program.context.set_arg(i, d);
}

void Kernel::set_arch(Arch arch) {
  TC_ASSERT(!compiled);
  this->arch = arch;
}

int Kernel::insert_arg(DataType dt, bool is_nparray) {
  args.push_back(Arg{dt, is_nparray, 0, false});
  return args.size() - 1;
}

TLANG_NAMESPACE_END
