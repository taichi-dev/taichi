#include "taichi/lang_util.h"
#include "opencl_program.h"
#include "opencl_kernel.h"
#include "taichi/program/program.h"

#include <vector>
#include <string>

// Discussion: https://stackoverflow.com/questions/28500496/opencl-function-found-deprecated-by-visual-studio
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>


TLANG_NAMESPACE_BEGIN
namespace opencl {

namespace {

std::string opencl_error(int err) {  // {{{
  switch (err) {
#define REG_ERRNO(x) case x: return #x; break;
  REG_ERRNO(CL_SUCCESS)
  REG_ERRNO(CL_DEVICE_NOT_FOUND)
  REG_ERRNO(CL_DEVICE_NOT_AVAILABLE)
  REG_ERRNO(CL_COMPILER_NOT_AVAILABLE)
  REG_ERRNO(CL_MEM_OBJECT_ALLOCATION_FAILURE)
  REG_ERRNO(CL_OUT_OF_RESOURCES)
  REG_ERRNO(CL_OUT_OF_HOST_MEMORY)
  REG_ERRNO(CL_PROFILING_INFO_NOT_AVAILABLE)
  REG_ERRNO(CL_MEM_COPY_OVERLAP)
  REG_ERRNO(CL_IMAGE_FORMAT_MISMATCH)
  REG_ERRNO(CL_IMAGE_FORMAT_NOT_SUPPORTED)
  REG_ERRNO(CL_BUILD_PROGRAM_FAILURE)
  REG_ERRNO(CL_MAP_FAILURE)
  REG_ERRNO(CL_MISALIGNED_SUB_BUFFER_OFFSET)
  REG_ERRNO(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
  REG_ERRNO(CL_COMPILE_PROGRAM_FAILURE)
  REG_ERRNO(CL_LINKER_NOT_AVAILABLE)
  REG_ERRNO(CL_LINK_PROGRAM_FAILURE)
  REG_ERRNO(CL_DEVICE_PARTITION_FAILED)
  REG_ERRNO(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
  REG_ERRNO(CL_INVALID_VALUE)
  REG_ERRNO(CL_INVALID_DEVICE_TYPE)
  REG_ERRNO(CL_INVALID_PLATFORM)
  REG_ERRNO(CL_INVALID_DEVICE)
  REG_ERRNO(CL_INVALID_CONTEXT)
  REG_ERRNO(CL_INVALID_QUEUE_PROPERTIES)
  REG_ERRNO(CL_INVALID_COMMAND_QUEUE)
  REG_ERRNO(CL_INVALID_HOST_PTR)
  REG_ERRNO(CL_INVALID_MEM_OBJECT)
  REG_ERRNO(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
  REG_ERRNO(CL_INVALID_IMAGE_SIZE)
  REG_ERRNO(CL_INVALID_SAMPLER)
  REG_ERRNO(CL_INVALID_BINARY)
  REG_ERRNO(CL_INVALID_BUILD_OPTIONS)
  REG_ERRNO(CL_INVALID_PROGRAM)
  REG_ERRNO(CL_INVALID_PROGRAM_EXECUTABLE)
  REG_ERRNO(CL_INVALID_KERNEL_NAME)
  REG_ERRNO(CL_INVALID_KERNEL_DEFINITION)
  REG_ERRNO(CL_INVALID_KERNEL)
  REG_ERRNO(CL_INVALID_ARG_INDEX)
  REG_ERRNO(CL_INVALID_ARG_VALUE)
  REG_ERRNO(CL_INVALID_ARG_SIZE)
  REG_ERRNO(CL_INVALID_KERNEL_ARGS)
  REG_ERRNO(CL_INVALID_WORK_DIMENSION)
  REG_ERRNO(CL_INVALID_WORK_GROUP_SIZE)
  REG_ERRNO(CL_INVALID_WORK_ITEM_SIZE)
  REG_ERRNO(CL_INVALID_GLOBAL_OFFSET)
  REG_ERRNO(CL_INVALID_EVENT_WAIT_LIST)
  REG_ERRNO(CL_INVALID_EVENT)
  REG_ERRNO(CL_INVALID_OPERATION)
  REG_ERRNO(CL_INVALID_GL_OBJECT)
  REG_ERRNO(CL_INVALID_BUFFER_SIZE)
  REG_ERRNO(CL_INVALID_MIP_LEVEL)
  REG_ERRNO(CL_INVALID_GLOBAL_WORK_SIZE)
  REG_ERRNO(CL_INVALID_PROPERTY)
  REG_ERRNO(CL_INVALID_IMAGE_DESCRIPTOR)
  REG_ERRNO(CL_INVALID_COMPILER_OPTIONS)
  REG_ERRNO(CL_INVALID_LINKER_OPTIONS)
  REG_ERRNO(CL_INVALID_DEVICE_PARTITION_COUNT)
  REG_ERRNO(CL_INVALID_PIPE_SIZE)
  REG_ERRNO(CL_INVALID_DEVICE_QUEUE)
  REG_ERRNO(CL_INVALID_SPEC_ID)
  REG_ERRNO(CL_MAX_SIZE_RESTRICTION_EXCEEDED)
  default: return fmt::format("{}", err);
  }
}  // }}}

std::vector<cl_platform_id> get_opencl_platforms() {
  cl_uint n;
  cl_int err = clGetPlatformIDs(0, NULL, &n);
  if (err < 0) {
    TI_ERROR("Failed to get OpenCL platforms: {}", err);
  }

  std::vector<cl_platform_id> platforms(n);
  clGetPlatformIDs(platforms.size(), platforms.data(), NULL);
  return platforms;
}


std::vector<cl_device_id> get_opencl_devices(cl_platform_id platform) {
  cl_uint n;
  cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &n);
  if (err < 0) {
    TI_ERROR("Failed to get OpenCL devices: {}", opencl_error(err));
  }

  std::vector<cl_device_id> devices(n);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices.size(), devices.data(), NULL);
  return devices;
}


std::string get_platform_info(cl_platform_id platform, cl_uint index) {
  size_t n;
  cl_int err = clGetPlatformInfo(platform, index, 0, NULL, &n);
  if (err < 0) {
    TI_ERROR("Failed to get OpenCL platform info: {}", opencl_error(err));
  }

  char s[n + 1];
  memset((void *)s, 0, n + 1);
  clGetPlatformInfo(platform, index, n, s, NULL);
  return std::string(s);
}


std::string get_device_info(cl_device_id device, cl_uint index) {
  size_t n;
  cl_int err = clGetDeviceInfo(device, index, 0, NULL, &n);
  if (err < 0) {
    TI_ERROR("Failed to get OpenCL device info: {}", opencl_error(err));
  }

  char s[n + 1];
  memset((void *)s, 0, n + 1);
  err = clGetDeviceInfo(device, index, n, s, NULL);
  if (err < 0) {
    TI_ERROR("Failed to get OpenCL device info string: {}", opencl_error(err));
  }
  return std::string(s);
}


template <typename T>
T get_device_info(cl_device_id device, cl_uint index) {
  T value = 0;
  cl_int err = clGetDeviceInfo(device, index, sizeof(T), &value, NULL);
  if (err < 0) {
    TI_ERROR("Failed to get OpenCL device info integer: {}", opencl_error(err));
  }
  return value;
}


void show_platforms_info_yaml(std::vector<cl_platform_id> platforms) {
  std::cout << "platforms:" << std::endl;
  for (auto pl: platforms) {
    std::cout << "- name: " << get_platform_info(pl, CL_PLATFORM_NAME) << std::endl;
    std::cout << "  vendor: " << get_platform_info(pl, CL_PLATFORM_VENDOR) << std::endl;
    std::cout << "  version: " << get_platform_info(pl, CL_PLATFORM_VERSION) << std::endl;

    auto devices = get_opencl_devices(platforms[0]);
    std::cout << "  devices:" << std::endl;
    for (auto de: devices) {
      std::cout << "  - name: " << get_device_info(de, CL_DEVICE_NAME) << std::endl;
      auto type = get_device_info<cl_device_type>(de, CL_DEVICE_TYPE);
      std::cout << "    device_type: " << (type == CL_DEVICE_TYPE_CPU ? "CPU" : type == CL_DEVICE_TYPE_GPU ? "GPU" : "DEFAULT") << std::endl;
      std::cout << "    vendor: " << get_device_info(de, CL_DEVICE_VENDOR) << std::endl;
      std::cout << "    version: " << get_device_info(de, CL_DEVICE_VERSION) << std::endl;
      std::cout << "    address_bits: " << get_device_info<int>(de, CL_DEVICE_ADDRESS_BITS) << std::endl;
      std::cout << "    local_mem_size: " << get_device_info<uint64_t>(de, CL_DEVICE_LOCAL_MEM_SIZE) << std::endl;
      std::cout << "    global_mem_size: " << get_device_info<uint64_t>(de, CL_DEVICE_GLOBAL_MEM_SIZE) << std::endl;
      std::cout << "    max_compute_units: " << get_device_info<uint64_t>(de, CL_DEVICE_MAX_COMPUTE_UNITS) << std::endl;
      std::cout << "    max_work_group_size: " << get_device_info<uint64_t>(de, CL_DEVICE_MAX_WORK_GROUP_SIZE) << std::endl;
      auto extensions = get_device_info(de, CL_DEVICE_EXTENSIONS);
      size_t pos = 0, new_pos;
      std::cout << "    extensions:" << std::endl;
      while (-1 != (new_pos = extensions.find(' ', pos))) {
        std::cout << "    - " << extensions.substr(pos, new_pos - pos) << std::endl;
        pos = new_pos + 1;
      }
    }
    std::cout << std::endl;
  }
}


struct CLContext {
  cl_context context;
  cl_command_queue cmdqueue;

  cl_device_id device;
  cl_platform_id platform;

  CLContext(int plat, int dev) {
    auto platforms = get_opencl_platforms();
    show_platforms_info_yaml(platforms);

    platform = platforms[plat];
    auto devices = get_opencl_devices(platform);
    device = devices[dev];

    cl_int err = -1;
    context = clCreateContext(NULL, devices.size(), devices.data(),
        NULL, NULL, &err);
    if (err < 0) {
      TI_ERROR("Failed to create OpenCL context: {}", opencl_error(err));
    }

    err = -1;
    cmdqueue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0) {
      TI_ERROR("Failed to create OpenCL command queue: {}", opencl_error(err));
    }
  }

  ~CLContext() {
    clReleaseContext(context);
    clReleaseCommandQueue(cmdqueue);
  }
};


struct CLBuffer {
  cl_mem buf;
  CLContext *ctx;
  size_t size;

  CLBuffer(CLContext *ctx, size_t size, cl_uint type = CL_MEM_READ_WRITE)
    : ctx(ctx), size(size) {
    cl_int err = -1;
    buf = clCreateBuffer(ctx->context, type, size, NULL, &err);
    if (err < 0) {
      TI_ERROR("Failed to create OpenCL buffer of size {}: {}",
          size, opencl_error(err));
    }
  }

  ~CLBuffer() {
    clReleaseMemObject(buf);
  }

  void write(const void *ptr, size_t size, size_t offset = 0) {
    cl_int err = clEnqueueWriteBuffer(ctx->cmdqueue, buf, CL_FALSE,
        offset, size, ptr, 0, NULL, NULL);
    if (err < 0) {
      TI_ERROR("Failed to write OpenCL buffer offset {} size {}: {}",
          offset, size, opencl_error(err));
    }
  }

  void read(void *ptr, size_t size, size_t offset = 0) {
    cl_int err = clEnqueueReadBuffer(ctx->cmdqueue, buf, CL_TRUE,
        offset, size, ptr, 0, NULL, NULL);
    if (err < 0) {
      TI_ERROR("Failed to read OpenCL buffer offset {} size {}: {}",
          offset, size, opencl_error(err));
    }
  }
};

struct CLProgram {
  CLContext *ctx;
  cl_program prog;

  CLProgram(CLContext *ctx, const std::string &src, std::string options = "")
    : ctx(ctx) {
    TI_DEBUG("Compiling OpenGL program:\n{}", src);

    cl_int err = -1;
    const char *src_ptr[1] = {src.c_str()};
    prog = clCreateProgramWithSource(ctx->context, 1, src_ptr, NULL, &err);
    if (err < 0) {
      TI_ERROR("Failed to create OpenCL program: {}", opencl_error(err));
    }

    err = clBuildProgram(prog, 0, NULL, options.c_str(), NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
      size_t n;
      clGetProgramBuildInfo(prog, ctx->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &n);
      char log[n + 1];
      clGetProgramBuildInfo(prog, ctx->device, CL_PROGRAM_BUILD_LOG, n, (void *)log, NULL);
      log[n] = 0;
      TI_ERROR("Failed to compile OpenCL program:\n{}", std::string(log));
    }

    if (err < 0) {
      TI_ERROR("Failed to build OpenCL program: {}", opencl_error(err));
    }
  }

  ~CLProgram() {
    clReleaseProgram(prog);
  }
};

struct CLKernel {
  CLContext *ctx;
  CLProgram *prog;
  cl_kernel kern;
  std::string name;

  CLKernel(CLProgram *prog, std::string name)
    : ctx(prog->ctx), prog(prog), name(name) {
    cl_int err = -1;
    kern = clCreateKernel(prog->prog, name.c_str(), &err);
    if (err < 0) {
      TI_ERROR("Failed to create OpenCL kernel {}: {}", name, opencl_error(err));
    }
  }

  ~CLKernel() {
    clReleaseKernel(kern);
  }

  template <typename T>
  void set_arg(int index, const T &value) {
    set_arg(index, &value, sizeof(T));
  }

  template <typename T>
  void set_arg(int index, const std::vector<T> &value) {
    set_arg(index, value.data(), value.size() * sizeof(T));
  }

  void set_arg(int index, const void *base, size_t size) {
    cl_int err = clSetKernelArg(kern, index, size, base);
    if (err < 0) {
      TI_ERROR("Failed to set kernel argument {} for {}: {}",
          index, name, opencl_error(err));
    }
  }

  void set_arg_buffer(int index, const CLBuffer &buf) {
    set_arg(index, &buf.buf, sizeof(cl_mem));
  }

  void launch(size_t n_dims, const size_t *global_dim, const size_t *block_dim) {
    cl_int err = clEnqueueNDRangeKernel(ctx->cmdqueue, kern,
        n_dims, NULL, global_dim, block_dim, 0, NULL, NULL);
    if (err < 0) {
      TI_ERROR("Failed to launch OpenCL kernel {} on ({}, {}): {}", name,
          n_dims ? global_dim[0] : 0, n_dims && block_dim ? block_dim[0] : 0,
          opencl_error(err));
    }
  }

  void launch(size_t global_dim, size_t block_dim = 0) {
    return launch(1, &global_dim, block_dim ? &block_dim : NULL);
  }
};

}  // namespace

struct OpenclProgram::Impl {
  std::unique_ptr<CLContext> context;
  std::unique_ptr<CLBuffer> root_buf;
  std::unique_ptr<CLBuffer> gtmp_buf;

  Impl(Program *prog) {
    context = std::make_unique<CLContext>(0, 0);
  }

  void allocate_root_buffer(size_t size) {
    root_buf = std::make_unique<CLBuffer>(context.get(), size);
    gtmp_buf = std::make_unique<CLBuffer>(context.get(),
        taichi_global_tmp_buffer_size);
  }
};

OpenclProgram::OpenclProgram(Program *prog)
  : impl(std::make_unique<Impl>(prog)) {
}

void OpenclProgram::allocate_root_buffer() {
  impl->allocate_root_buffer(layout_size);
}

OpenclProgram::~OpenclProgram() = default;

struct OpenclKernel::Impl {
  OpenclProgram *prog;

  std::unique_ptr<CLProgram> program;
  std::vector<std::tuple<OpenclOffloadMeta, std::unique_ptr<CLKernel>>> kernels;

  Kernel *kernel;

  Impl(OpenclProgram *prog, Kernel *kernel,
      std::vector<OpenclOffloadMeta> const &offloads,
      std::string const &source)
    : prog(prog), kernel(kernel) {
      program = std::make_unique<CLProgram>(prog->impl->context.get(), source);

      for (auto const &meta: offloads) {
        auto ker = std::make_unique<CLKernel>(program.get(), meta.kernel_name);
        kernels.push_back(std::make_tuple(meta, std::move(ker)));
      }
  }

  void launch(Context *ctx) {
    TI_ASSERT(prog->impl->root_buf);

    std::vector<std::unique_ptr<CLBuffer>> extrs;

    for (int i = 0; i < kernel->args.size(); i++) {
      if (kernel->args[i].is_nparray) {
        auto extr = std::make_unique<CLBuffer>(prog->impl->context.get(),
            kernel->args[i].size);
        extr->write(reinterpret_cast<void *>(ctx->args[i]), kernel->args[i].size);
        extrs.push_back(std::move(extr));
      }
    }

    for (const auto &[meta, ker]: kernels) {
      ker->set_arg_buffer(0, *prog->impl->root_buf);
      ker->set_arg_buffer(1, *prog->impl->gtmp_buf);

      auto extr_iter = extrs.begin();
      for (int i = 0; i < kernel->args.size(); i++) {
        if (kernel->args[i].is_nparray) {
          auto const &extr = *extr_iter++;
          ker->set_arg_buffer(i + 2, *extr);
          ker->set_arg(i + 2 + kernel->args.size(),
              ctx->extra_args[i], sizeof(int) * 8);

        } else {
          ker->set_arg(i + 2, &ctx->args[i], data_type_size(kernel->args[i].dt));
        }
      }

      ker->launch(meta.global_dim, meta.block_dim);
    }

    if (kernel->rets.size()) {
      TI_ASSERT(kernel->program.result_buffer);
      prog->impl->gtmp_buf->read(kernel->program.result_buffer,
          kernel->rets.size() * sizeof(uint64_t));
    }

    auto extr_iter = extrs.begin();
    for (int i = 0; i < kernel->args.size(); i++) {
      if (kernel->args[i].is_nparray) {
        auto const &extr = *extr_iter++;
        extr->read(reinterpret_cast<void *>(ctx->args[i]), kernel->args[i].size);
      }
    }
  }
};

OpenclKernel::OpenclKernel(OpenclProgram *prog, Kernel *kernel,
    std::vector<OpenclOffloadMeta> const &offloads, std::string const &source)
  : name(kernel->name), source(source), impl(std::make_unique<Impl>(prog,
        kernel, offloads, source)) {
}

OpenclKernel::~OpenclKernel() = default;

void OpenclKernel::launch(Context *ctx) {
  impl->launch(ctx);
}

}  // namespace opencl
TLANG_NAMESPACE_END
