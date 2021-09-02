#pragma once

#include "taichi/common/core.h"

#include <iostream>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <assert.h>

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

TLANG_NAMESPACE_BEGIN

class Kernel;

namespace dx {

const int taichi_dx_earg_base = 64;  // 8x8; same as taichi_opengl_earg_base

struct HLSLLauncher;

HRESULT CreateComputeDevice(ID3D11Device **device,
                            ID3D11DeviceContext **context,
                            bool force_ref);

bool is_dx_api_available();

void dump_buffers();

struct ParallelSize {
 public:
  size_t block_dim;
  size_t grid_dim;
  ParallelSize(size_t b = 1, size_t g = 1) : block_dim(b), grid_dim(g) {
  }
};

// Mostly copied from opengl_api.h
struct CompiledKernel {
  struct Impl {
    std::string kernel_name;
    std::unique_ptr<ParallelSize> ps;
    std::string source;
    ID3D11ComputeShader* compute_shader; // Opaque object
    Impl(const std::string &kernel_name,
         const std::string &kernel_source_code,
         std::unique_ptr<ParallelSize> ps_);
    void dispatch_compute(HLSLLauncher *launcher) const;
  };
  std::unique_ptr<Impl> impl;

  CompiledKernel(CompiledKernel &&) = default;
  CompiledKernel &operator=(CompiledKernel &&) = default;

  CompiledKernel(const std::string &kernel_name_,
                 const std::string &kernel_source_code,
                 std::unique_ptr<ParallelSize> ps_);
  ~CompiledKernel() = default;

  void dispatch_compute(HLSLLauncher *launcher) const;
};

enum ReturnBufferId {
  data_i32,
  data_f32,
  extr_i32,
  extr_f32,
};

struct CompiledProgram {
  struct Impl {
    std::vector<std::unique_ptr<CompiledKernel>> kernels;
    int arg_count, ret_count;
    std::map<int, size_t> ext_arr_map;
    ReturnBufferId return_buffer_id;
    Impl(Kernel *kernel);
    void add(const std::string &kernel_name,
             const std::string &kernel_source_code,
             std::unique_ptr<ParallelSize> ps);
    void launch(Context &ctx, HLSLLauncher *launcher) const;
  };

  std::unique_ptr<Impl> impl;
  CompiledProgram(Kernel *kernel);
  void add(const std::string &kernel_name,
           const std::string &kernel_source_code,
           std::unique_ptr<ParallelSize> ps);
  void launch(Context &ctx, HLSLLauncher *launcher) const;
};


struct HLSLLauncherImpl {
  std::vector<std::unique_ptr<CompiledProgram>> programs;
};

struct HLSLLauncher {
  std::unique_ptr<HLSLLauncherImpl> impl;
  HLSLLauncher(size_t size);
  ~HLSLLauncher() = default;
  void *result_buffer;
  void keep(std::unique_ptr<CompiledProgram> program);
};

}  // namespace dx

TLANG_NAMESPACE_END
