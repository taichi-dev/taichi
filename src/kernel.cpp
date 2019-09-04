#include <taichi/common/task.h>
#include <taichi/taichi>
#include "program.h"
#if defined(CUDA_FOUND)
#include <cuda_runtime.h>
#endif

TLANG_NAMESPACE_BEGIN

Kernel::Kernel(Program &program,
               std::function<void()> func,
               std::string name,
               bool grad)
    : program(program), name(name), grad(grad) {
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
  std::vector<void *> host_buffers(args.size());
  std::vector<void *> device_buffers(args.size());
  if (program.config.arch == Arch::gpu) {
#if defined(CUDA_FOUND)
    // copy data to GRAM
    bool has_buffer = false;
    for (int i = 0; i < (int)args.size(); i++) {
      if (args[i].is_nparray) {
        has_buffer = true;
        cudaMalloc(&device_buffers[i], args[i].size);
        // replace host buffer with device buffer
        host_buffers[i] = program.context.get_arg<void *>(i);
        set_arg_nparray(i, (uint64)device_buffers[i], args[i].size);
        cudaMemcpy(device_buffers[i], host_buffers[i], args[i].size,
                   cudaMemcpyHostToDevice);
      } }
    if (has_buffer)
      cudaDeviceSynchronize();
    auto c = program.get_context();
    compiled(c);
    if (has_buffer)
      cudaDeviceSynchronize();
    for (int i = 0; i < (int)args.size(); i++) {
      if (args[i].is_nparray) {
        cudaMemcpy(host_buffers[i], device_buffers[i], args[i].size,
                   cudaMemcpyDeviceToHost);
        cudaFree(device_buffers[i]);
      }
    }
#else
    TC_ERROR("No CUDA");
#endif
  } else {
    auto c = program.get_context();
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

void Kernel::set_arg_nparray(int i, uint64 d, uint64 size) {
  TC_ASSERT_INFO(args[i].is_nparray,
                 "Setting numpy array to scalar argument is not allowed");
  args[i].size = size;
  program.context.set_arg(i, d);
}

TLANG_NAMESPACE_END
