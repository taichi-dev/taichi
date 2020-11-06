#include "taichi/lang_util.h"
#include "opencl_program.h"
#include "opencl_kernel.h"

#include <vector>
#include <string>

// Discussion: https://stackoverflow.com/questions/28500496/opencl-function-found-deprecated-by-visual-studio
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>


TLANG_NAMESPACE_BEGIN
namespace opencl {

namespace {

std::vector<cl_platform_id> get_opencl_platforms() {
  cl_uint n;
  cl_int err = clGetPlatformIDs(0, NULL, &n);
  if (err < 0) {
    TI_ERROR("Failed to get OpenCL platforms: error {}", err);
  }

  std::vector<cl_platform_id> platforms(n);
  clGetPlatformIDs(platforms.size(), platforms.data(), NULL);
  return platforms;
}


std::vector<cl_device_id> get_opencl_devices(cl_platform_id platform) {
  cl_uint n;
  cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &n);
  if (err < 0) {
    TI_ERROR("Failed to get OpenCL devices: error {}", err);
  }

  std::vector<cl_device_id> devices(n);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices.size(), devices.data(), NULL);
  return devices;
}


std::string get_platform_info(cl_platform_id platform, cl_uint index) {
  size_t n;
  cl_int err = clGetPlatformInfo(platform, index, 0, NULL, &n);
  if (err < 0) {
    TI_ERROR("Failed to get OpenCL platform info: error {}", err);
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
    TI_ERROR("Failed to get OpenCL device info: error {}", err);
  }

  char s[n + 1];
  memset((void *)s, 0, n + 1);
  clGetDeviceInfo(device, index, n, s, NULL);
  return std::string(s);
}


void show_platforms_info_yaml(std::vector<cl_platform_id> platforms) {
  std::cout << "platforms:" << std::endl;
  for (auto pl: platforms) {
    std::cout << "- name: " << get_platform_info(pl, CL_PLATFORM_NAME) << std::endl;
    std::cout << "  vendor: " << get_platform_info(pl, CL_PLATFORM_VENDOR) << std::endl;
    std::cout << "  version: " << get_platform_info(pl, CL_PLATFORM_VERSION) << std::endl;
    std::cout << "  extensions: " << get_platform_info(pl, CL_PLATFORM_EXTENSIONS) << std::endl;

    auto devices = get_opencl_devices(platforms[0]);
    std::cout << "  devices:" << std::endl;
    for (auto de: devices) {
      std::cout << "  - name: " << get_device_info(de, CL_DEVICE_NAME) << std::endl;
      std::cout << "    vendor: " << get_device_info(de, CL_DEVICE_VENDOR) << std::endl;
      std::cout << "    version: " << get_device_info(de, CL_DEVICE_VERSION) << std::endl;
      std::cout << "    extensions: " << get_device_info(de, CL_DEVICE_EXTENSIONS) << std::endl;
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

    cl_int err;
    context = clCreateContext(NULL, devices.size(), devices.data(), NULL, NULL, &err);
    if (err < 0) {
      TI_ERROR("Failed to create OpenCL context: error {}", err);
    }

    cmdqueue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0) {
      TI_ERROR("Failed to create OpenCL command queue: error {}", err);
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
    cl_int err;
    buf = clCreateBuffer(ctx->context, type, size, NULL, &err);
    if (err < 0) {
      TI_ERROR("Failed to create OpenCL buffer: error {}", err);
    }
  }

  ~CLBuffer() {
    clReleaseMemObject(buf);
  }

  void write(const void *ptr, size_t size, size_t offset = 0) {
    cl_int err = clEnqueueWriteBuffer(ctx->cmdqueue, buf, CL_FALSE,
        offset, size, ptr, 0, NULL, NULL);
    if (err < 0) {
      std::cerr << err << std::endl;
      TI_ERROR("Failed to write OpenCL buffer: error {}", err);
    }
  }

  void read(void *ptr, size_t size, size_t offset = 0) {
    cl_int err = clEnqueueReadBuffer(ctx->cmdqueue, buf, CL_TRUE,
        offset, size, ptr, 0, NULL, NULL);
    if (err < 0) {
      std::cerr << err << std::endl;
      TI_ERROR("Failed to read OpenCL buffer: error {}", err);
    }
  }
};


struct CLProgram {
  CLContext *ctx;
  cl_program prog;

  CLProgram(CLContext *ctx, const std::string &src, std::string options = "")
    : ctx(ctx) {
    TI_DEBUG("Compiling OpenGL program:\n{}", src);

    cl_int err;
    const char *src_ptr[1] = {src.c_str()};
    prog = clCreateProgramWithSource(ctx->context, 1, src_ptr, NULL, &err);
    if (err < 0) {
      TI_ERROR("Failed to create OpenCL program: error {}", err);
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
      TI_ERROR("Failed to build OpenCL program: error {}", err);
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
    cl_int err;
    kern = clCreateKernel(prog->prog, name.c_str(), &err);
    if (err < 0) {
      TI_ERROR("Failed to create OpenCL kernel {}: error {}", name, err);
    }
  }

  ~CLKernel() {
    clReleaseKernel(kern);
  }

  template <typename T>
  void set_arg(int index, const T &value) {
    cl_int err = clSetKernelArg(kern, index, sizeof(T), &value);
    if (err < 0) {
      TI_ERROR("Failed to set kernel argument for {}: error {}", name, err);
    }
  }

  void set_arg(int index, const CLBuffer &buf) {
    cl_int err = clSetKernelArg(kern, index, sizeof(cl_mem), &buf.buf);
    if (err < 0) {
      TI_ERROR("Failed to set kernel argument for {}: error {}", name, err);
    }
  }

  void launch(size_t n_dims, const size_t *grid_dim, const size_t *block_dim) {
    cl_int err = clEnqueueNDRangeKernel(ctx->cmdqueue, kern,
        n_dims, NULL, grid_dim, block_dim, 0, NULL, NULL);
    if (err < 0) {
      TI_ERROR("Failed to launch OpenCL kernel {}: error {}", name, err);
    }
  }

  void launch(size_t grid_dim, size_t block_dim) {
    return launch(1, &grid_dim, &block_dim);
  }

  void launch(size_t grid_dim) {
    return launch(1, &grid_dim, NULL);
  }
};

}  // namespace

struct OpenclProgram::Impl {
  std::unique_ptr<CLContext> context;

  Impl(Program *prog) {
    context = std::make_unique<CLContext>(0, 0);
  }
};

OpenclProgram::OpenclProgram(Program *prog)
  : impl(std::make_unique<Impl>(prog)) {
}

OpenclProgram::~OpenclProgram() = default;

struct OpenclKernel::Impl {
  std::unique_ptr<CLProgram> program;
  std::vector<std::unique_ptr<CLKernel>> kernels;

  Impl(OpenclProgram *prog, int offload_count, std::string const &source) {
      program = std::make_unique<CLProgram>(prog->impl->context.get(), source);

      for (int i = 0; i < offload_count; i++) {
        kernels.push_back(std::make_unique<CLKernel>(
              program.get(), fmt::format("offload_{}", i)));
      }
  }

  void launch(Context *ctx) {
    TI_WARN("launch {}", (void *)ctx);
  }
};

OpenclKernel::OpenclKernel(OpenclProgram *prog, std::string name,
    int offload_count, std::string const &source)
  : name(name), source(source), impl(std::make_unique<Impl>(prog,
        offload_count, source)) {
}

OpenclKernel::~OpenclKernel() = default;

void OpenclKernel::launch(Context *ctx) {
  impl->launch(ctx);
}

}  // namespace opencl
TLANG_NAMESPACE_END
