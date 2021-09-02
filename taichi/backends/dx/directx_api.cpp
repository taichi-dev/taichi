#include "directx_api.h"
#include <chrono>
#include <filesystem>
#include "taichi/program/kernel.h"

TLANG_NAMESPACE_BEGIN

namespace dx {

ID3D11Device *g_device;
ID3D11DeviceContext *g_context;
ID3D11Buffer *g_args_i32_buf, *g_args_f32_buf,
  *g_data_i32_buf, *g_data_f32_buf,
  *g_extr_i32_buf, *g_extr_f32_buf,
  *g_locks_buf, *tmp_arg_buf;
ID3D11UnorderedAccessView *g_args_i32_uav, *g_args_f32_uav,
  *g_data_i32_uav, *g_data_f32_uav,
  *g_extr_i32_uav, *g_extr_f32_uav,
  *g_locks_uav;

HRESULT CreateComputeDevice(ID3D11Device **out_device,
                            ID3D11DeviceContext **out_context,
                            bool force_ref) {
  const D3D_FEATURE_LEVEL levels[] = {
      D3D_FEATURE_LEVEL_11_1,
      D3D_FEATURE_LEVEL_11_0,
      D3D_FEATURE_LEVEL_10_1,
      D3D_FEATURE_LEVEL_10_0,
  };

  UINT flags = 0;
  //flags |= D3D11_CREATE_DEVICE_DEBUG;

  ID3D11Device *device = nullptr;
  ID3D11DeviceContext *context = nullptr;
  HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                 flags, levels, _countof(levels),
                                 D3D11_SDK_VERSION, &device, nullptr, &context);

  if (FAILED(hr)) {
    printf("Failed to create D3D11 device: %08X\n", hr);
    return -1;
  }

  if (device->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0) {
    D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS hwopts = {0};
    device->CheckFeatureSupport(D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, &hwopts,
                                sizeof(hwopts));
    if (!hwopts.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x) {
      device->Release();
      printf(
          "DirectCompute not supported via "
          "ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4\n");
      return -1;
    }
  }

  *out_device = device;
  *out_context = context;
  return hr;
}

HRESULT CreateStructuredBuffer(ID3D11Device *device,
                               UINT element_size,
                               UINT count,
                               void *init_data,
                               ID3D11Buffer **out_buf) {
  *out_buf = nullptr;
  D3D11_BUFFER_DESC desc = {};
  desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
  desc.ByteWidth = element_size * count;
  desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
  desc.StructureByteStride = element_size;
  if (init_data) {
    D3D11_SUBRESOURCE_DATA data;
    data.pSysMem = init_data;
    return device->CreateBuffer(&desc, &data, out_buf);
  } else {
    return device->CreateBuffer(&desc, nullptr, out_buf);
  }
}


HRESULT CreateRawBuffer(ID3D11Device *device,
                        UINT size,
                        void *init_data,
                        ID3D11Buffer **out_buf) {
  *out_buf = nullptr;
  D3D11_BUFFER_DESC desc = {};
  desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
  desc.ByteWidth = size;
  desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
  if (init_data) {
    D3D11_SUBRESOURCE_DATA data;
    data.pSysMem = init_data;
    return device->CreateBuffer(&desc, &data, out_buf);
  } else {
    return device->CreateBuffer(&desc, nullptr, out_buf);
  }
}

HRESULT CreateBufferUAV(ID3D11Device *device,
                        ID3D11Buffer *buffer,
                        ID3D11UnorderedAccessView **out_uav) {
  D3D11_BUFFER_DESC buf_desc = {};
  buffer->GetDesc(&buf_desc);
  D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
  uav_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
  uav_desc.Buffer.FirstElement = 0;
  if (buf_desc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS) {
    uav_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    uav_desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    uav_desc.Buffer.NumElements = buf_desc.ByteWidth / 4;
  } else if (buf_desc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED) {
    uav_desc.Format = DXGI_FORMAT_UNKNOWN;
    uav_desc.Buffer.NumElements =
        buf_desc.ByteWidth / buf_desc.StructureByteStride;
  } else
    return E_INVALIDARG;
  return device->CreateUnorderedAccessView(buffer, &uav_desc, out_uav);
}

HRESULT CompileComputeShaderFromString(const std::string &source,
                                       LPCSTR entry_point,
                                       ID3D11Device *device,
                                       ID3DBlob **blob) {
  UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
  LPCSTR profile = (device->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_0)
                       ? "cs_5_0"
                       : "cs_4_0";
  const D3D_SHADER_MACRO defines[] = {"EXAMPLE_DEFINE", "1", NULL, NULL};
  ID3DBlob *shader_blob = nullptr, *error_blob = nullptr;
  HRESULT hr =
      D3DCompile(source.data(), source.size(), nullptr, defines, nullptr,
                 entry_point, profile, flags, 0, &shader_blob, &error_blob);
  if (FAILED(hr)) {
    printf("Error in CompileComputeShaderFromString\n");
    if (error_blob) {
      printf("%s\n", (char *)error_blob->GetBufferPointer());
      error_blob->Release();
    } else
      printf("error_blob is null\n");
    if (shader_blob) {
      shader_blob->Release();
    }
    fflush(stdout);
    return hr;
  }
  *blob = shader_blob;
  return hr;
}

char* DumpBuffer(ID3D11Buffer *buf, size_t* len) {
  D3D11_BUFFER_DESC desc = {};
  buf->GetDesc(&desc);
  ID3D11Buffer *tmpbuf;
  HRESULT hr;

  D3D11_BUFFER_DESC tmp_desc = {};
  tmp_desc.ByteWidth = desc.ByteWidth;
  tmp_desc.Usage = D3D11_USAGE_STAGING;
  tmp_desc.BindFlags = 0;
  tmp_desc.MiscFlags = 0;
  tmp_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  hr = g_device->CreateBuffer(&tmp_desc, nullptr, &tmpbuf);
  assert(SUCCEEDED(hr));
  g_context->CopyResource(tmpbuf, buf);

  D3D11_MAPPED_SUBRESOURCE mapped;
  hr = g_context->Map(tmpbuf, 0, D3D11_MAP_READ, 0, &mapped);
  assert(SUCCEEDED(hr));
  char *ret = new char[desc.ByteWidth];
  if (len) {
    *len = desc.ByteWidth;
  }
  memcpy(ret, mapped.pData, desc.ByteWidth);
  g_context->Unmap(tmpbuf, 0);
  tmpbuf->Release();
  return ret;
}

bool initialize_dx(bool error_tolerance = false) {
  if (g_device == nullptr || g_context == nullptr) {
    TI_TRACE("Creating D3D11 device");
    CreateComputeDevice(&g_device, &g_context, false);

    TI_TRACE("Creating D3D11 buffers");
    const int N = 1048576;
    CreateStructuredBuffer(g_device, 4, N, nullptr, &g_data_i32_buf);
    CreateStructuredBuffer(g_device, 4, N, nullptr, &g_data_f32_buf);
    CreateStructuredBuffer(g_device, 4, N, nullptr, &g_args_i32_buf);
    CreateStructuredBuffer(g_device, 4, N, nullptr, &g_args_f32_buf);
    CreateStructuredBuffer(g_device, 4, N, nullptr, &g_extr_i32_buf);
    CreateStructuredBuffer(g_device, 4, N, nullptr, &g_extr_f32_buf);
    char *zeroes = new char[N * 4];
    memset(zeroes, 0x00, N * 4);
    CreateRawBuffer(g_device, 4 * N, zeroes, &g_locks_buf);
    delete[] zeroes;

    TI_TRACE("Creating D3D11 UAVs");
    CreateBufferUAV(g_device, g_data_i32_buf, &g_data_i32_uav);
    CreateBufferUAV(g_device, g_data_f32_buf, &g_data_f32_uav);
    CreateBufferUAV(g_device, g_args_i32_buf, &g_args_i32_uav);
    CreateBufferUAV(g_device, g_args_f32_buf, &g_args_f32_uav);
    CreateBufferUAV(g_device, g_extr_i32_buf, &g_extr_i32_uav);
    CreateBufferUAV(g_device, g_extr_f32_buf, &g_extr_f32_uav);
    CreateBufferUAV(g_device, g_locks_buf, &g_locks_uav);

    
  // Copy to the UAVs
    D3D11_BUFFER_DESC desc;
    g_args_f32_buf->GetDesc(&desc);
    D3D11_BUFFER_DESC tmp_desc = {};
    tmp_desc.ByteWidth = desc.ByteWidth;
    tmp_desc.BindFlags = 0;
    tmp_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE | D3D11_CPU_ACCESS_READ;
    tmp_desc.MiscFlags = 0;
    tmp_desc.Usage = D3D11_USAGE_STAGING;
    HRESULT hr = g_device->CreateBuffer(&tmp_desc, nullptr, &tmp_arg_buf);
    assert(SUCCEEDED(hr));

  } else {
    TI_TRACE("D3D11 device has already been created.");
  }
  return true;
}

bool is_dx_api_available() {
	//if (get_environ_config("TI_ENABLE_DX", 1) == 0)
	//	return false;
    return initialize_dx();
}

// CompiledKernel, CompiledKernel::Impl

CompiledKernel::CompiledKernel(const std::string &kernel_name_,
                               const std::string &kernel_source_code,
                               std::unique_ptr<ParallelSize> ps_) :
   impl(std::make_unique<Impl>(kernel_name_, kernel_source_code,
     std::move(ps_))) {
}

void CompiledKernel::dispatch_compute(HLSLLauncher *launcher) const {
  impl->dispatch_compute(launcher);
}

CompiledKernel::Impl::Impl(const std::string &kernel_name,
                           const std::string &kernel_source_code,
                           std::unique_ptr<ParallelSize> ps_)
    : kernel_name(kernel_name), ps(std::move(ps_)), compute_shader(nullptr) {
  printf("CompiledKernel::Impl ctor\n");
  printf("kernel_name: %s\n", kernel_name.c_str());
  printf("kernel_source_code: %s\n", kernel_source_code.c_str());
  // todo: add dimension limit

  // Build program here
  ID3DBlob *shader_blob;
  HRESULT hr = CompileComputeShaderFromString(kernel_source_code, "CSMain",
                                              g_device, &shader_blob);
  if (SUCCEEDED(hr)) {
    TI_TRACE("Kernel compilation OK");

    ID3D11ComputeShader *cs = nullptr;
    hr = g_device->CreateComputeShader(shader_blob->GetBufferPointer(),
                                       shader_blob->GetBufferSize(), nullptr,
                                       &cs);
    shader_blob->Release();
    compute_shader = cs;
    if (SUCCEEDED(hr)) {
      TI_TRACE("Create Compute Shader OK");
    }

  } else {
    TI_ERROR("Kernel compilation error");
  }
}

void CompiledKernel::Impl::dispatch_compute(HLSLLauncher *launcher) const {
  // 1. set shader
  bool should_print = false;

  char *x = getenv("VERBOSE");
  if (x && std::atoi(x) == 1) {
    should_print = true;
  }

  if (1) {
    TI_TRACE("dispatch_compute<<<{},{}>>>", ps->grid_dim, ps->block_dim);
  }

  // debug
  size_t nbytes;
  float *f32_data0 = nullptr;
  
  if (should_print) {
    f32_data0 = (float *)(DumpBuffer(g_data_f32_buf, &nbytes));
  }

  // Temporary u0=_data_i32_, u1=_data_f32_, u2=_args_i32_, u3=_args_f32_
  ID3D11UnorderedAccessView *uavs[] = {g_data_i32_uav, g_data_f32_uav,
                                       g_args_i32_uav, g_args_f32_uav,
                                       g_extr_i32_uav, g_extr_f32_uav,
                                       g_locks_uav};
  g_context->CSSetShader(compute_shader, nullptr, 0);
  g_context->CSSetUnorderedAccessViews(0, 7, uavs, nullptr);

  g_context->Dispatch(ps->grid_dim, 1, 1);
  // 2. memory barrier


  
}

// CompiledProgram, CompiledProgram::Impl

CompiledProgram::CompiledProgram(Kernel *kernel)
    : impl(std::make_unique<Impl>(kernel)) {
}

void CompiledProgram::Impl::add(const std::string &kernel_name,
                                const std::string &kernel_source_code,
                                std::unique_ptr<ParallelSize> ps) {
  TI_TRACE("CompiledProgram::Impl::add");
  kernels.push_back(std::make_unique<CompiledKernel>(
      kernel_name, kernel_source_code, std::move(ps)));
}

CompiledProgram::Impl::Impl(Kernel *kernel) {
  arg_count = kernel->args.size();
  ret_count = kernel->rets.size();

  for (int i = 0; i < arg_count; i++) {
    if (kernel->args[i].is_external_array) { // xv = ti.Vector.field(2, float, 128)
      ext_arr_map[i] = kernel->args[i].size;
    }
  }
}

// args UAVs are used both for arguments and return values
void CompiledProgram::Impl::launch(Context &ctx, HLSLLauncher *launcher) const {
  std::chrono::time_point<std::chrono::steady_clock> t0 =
      std::chrono::steady_clock::now();

  bool should_print = false;
  char *x = getenv("VERBOSE");
  if (x && std::atoi(x) > 0) {
    should_print = true;
  } else {
    /*
    if (this->kernels[0]->impl->kernel_name != "kernel_7") {
      should_print = true;
    }*/
  }

  if (1) {
    printf("CompiledProgram launch, %llu kernels in total:\n", kernels.size());
    int i = 0;
    for (const auto &kernel : kernels) {
      printf("%d. %s\n", i, kernel->impl->kernel_name.c_str());
      i++;
    }
    TI_TRACE("ctx.args: {} {} {} {} {} {} {} {}, arg_count={}, ret_count={}",
             ctx.args[0], ctx.args[1], ctx.args[2], ctx.args[3], ctx.args[4],
             ctx.args[5], ctx.args[6], ctx.args[7], arg_count, ret_count);

    TI_TRACE("ctx.extra_args: {} {} {} {} {} {} {} {}", ctx.extra_args[0][0],
             ctx.extra_args[0][1], ctx.extra_args[0][2], ctx.extra_args[0][3],
             ctx.extra_args[0][4], ctx.extra_args[0][5], ctx.extra_args[0][6],
             ctx.extra_args[0][7]);
  }
  
  std::vector<char> args;
  args.resize(std::max(arg_count, ret_count) * sizeof(uint64_t));

  // TODO: add support for read extr buffer
  // TODO: add support for writing >1 extr buffers
  void *extptr = nullptr;
  size_t extsize = 256; // Should be at least this much for a return buffer

  if (ext_arr_map.size()) {
    const size_t extra_size = arg_count * size_t(taichi_max_num_indices) * sizeof(uint64);
    args.resize(taichi_dx_earg_base + extra_size);
    std::memcpy(args.data() + taichi_dx_earg_base, 
      ctx.extra_args,
      extra_size);
    if (ext_arr_map.size() == 1) {
      extptr = (void *)ctx.args[ext_arr_map.begin()->first];
      extsize = ext_arr_map.begin()->second;
      ctx.args[ext_arr_map.begin()->first] = 0; // otherwise all 0s
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  // Reinterpret as uint64s
  // Todo: check if the following is correct:
  // [0]~[7] are args, [16]+ are extra args, total 24
  constexpr int LEN = 24;
  int int_args[LEN];
  for (int i = 0; i < 8; i++) {
    int_args[i] = ctx.get_arg<int>(i);
  }
  // FIXME: the following is a short-circuit solution
  for (int i = 0; i < 8; i++) {
    int idx = taichi_dx_earg_base / 4 + i;
    int_args[idx] = ctx.extra_args[0][i];
  }
  D3D11_MAPPED_SUBRESOURCE mapped;
  HRESULT hr = g_context->Map(tmp_arg_buf, 0, D3D11_MAP_WRITE, 0, &mapped);
  assert(SUCCEEDED(hr));
  memcpy(mapped.pData, int_args, sizeof(int_args));
  g_context->Unmap(tmp_arg_buf, 0);
  D3D11_BOX copy_range = {0};
  copy_range.left = 0;
  copy_range.right = LEN * sizeof(int);
  copy_range.top = 0;
  copy_range.bottom = 1;
  copy_range.front = 0;
  copy_range.back = 1;
  g_context->CopySubresourceRegion(g_args_i32_buf, 0, 0, 0, 0, tmp_arg_buf, 0,
                                   &copy_range);

  float float_args[8];
  for (int i = 0; i < 8; i++) {
    float_args[i] = ctx.get_arg<float>(i);
  }

  hr = g_context->Map(tmp_arg_buf, 0, D3D11_MAP_WRITE, 0, &mapped);
  assert(SUCCEEDED(hr));
  memcpy(mapped.pData, float_args, sizeof(float_args));
  g_context->Unmap(tmp_arg_buf, 0);
  copy_range.right = 8 * sizeof(float);
  g_context->CopySubresourceRegion(g_args_f32_buf, 0, 0, 0, 0, tmp_arg_buf, 0,
                                   &copy_range);

  std::chrono::time_point<std::chrono::steady_clock> t1 =
      std::chrono::steady_clock::now();
  for (const auto &kernel : kernels) {
    kernel->dispatch_compute(launcher);
  }
  std::chrono::time_point<std::chrono::steady_clock> t2 = 
      std::chrono::steady_clock::now();

  // Process return values
  // Very crappy for now

  // TODO: specify a correct return size for non-ext copies
  copy_range.right = extsize;

  if (should_print) {
    switch (return_buffer_id) {
      case data_i32:
        TI_TRACE("DXX return value is in data_i32 buf\n");
        break;
      case data_f32:
        TI_TRACE("DXX return value is in data_f32 buf\n");
        break;
      case extr_i32:
        TI_TRACE("DXX return value is in extr_i32 buf\n");
        break;
      case extr_f32:
        TI_TRACE("DXX return value is in extr_f32 buf\n");
        break;
    } 
  }

  switch (return_buffer_id) {
    case data_i32:
      g_context->CopySubresourceRegion(tmp_arg_buf, 0, 0, 0, 0, g_args_i32_buf,
                                       0, &copy_range);
      break;
    case data_f32:
      g_context->CopySubresourceRegion(tmp_arg_buf, 0, 0, 0, 0, g_args_f32_buf,
                                       0, &copy_range);
      break;
    case extr_i32:
      g_context->CopySubresourceRegion(tmp_arg_buf, 0, 0, 0, 0, g_extr_i32_buf,
                                       0, &copy_range);
      break;
    case extr_f32:
      g_context->CopySubresourceRegion(tmp_arg_buf, 0, 0, 0, 0, g_extr_f32_buf,
                                       0, &copy_range);
      break;
  }

  hr = g_context->Map(tmp_arg_buf, 0, D3D11_MAP_READ, 0, &mapped);
  assert(SUCCEEDED(hr));
  memcpy(float_args, mapped.pData, sizeof(float_args));

  if (extptr) {
    memcpy(extptr, mapped.pData, extsize);
  }

  uint64_t *ptr = (uint64_t*)(launcher->result_buffer);
  for (int i = 0; i < ret_count; i++) {
    ptr[i] = *(reinterpret_cast<uint64_t*>(&float_args[i]));
  }
  g_context->Unmap(tmp_arg_buf, 0);

  std::chrono::time_point<std::chrono::steady_clock> t3 =
      std::chrono::steady_clock::now();
  int ms_all =
      std::chrono::duration_cast<std::chrono::microseconds>(t3 - t0).count();
  int ms_kernels =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  if (should_print) {
    printf("Kernel %s, %g ms all, %g ms kernel-only\n",
           this->kernels[0]->impl->kernel_name.c_str(), ms_all / 1000.0f,
           ms_kernels / 1000.0f);
  }
}

void CompiledProgram::add(const std::string &kernel_name,
  const std::string &kernel_source_code,
  std::unique_ptr<ParallelSize> ps) {
  impl->add(kernel_name, kernel_source_code, std::move(ps));
}

void CompiledProgram::launch(Context &ctx, HLSLLauncher *launcher) const {
  impl->launch(ctx, launcher);
}

HLSLLauncher::HLSLLauncher(size_t size) {
  initialize_dx();
  TI_TRACE("HLSLLauncher ctor");
  impl = std::make_unique<HLSLLauncherImpl>();
}

void HLSLLauncher::keep(std::unique_ptr<CompiledProgram> program) {
  impl->programs.push_back(std::move(program));
}

void dump_buffers() {
  printf("[dx backend] Dump buffers, CWD: %ls\n",
    std::filesystem::current_path().c_str());

  const char *names[] = {"args_i32", "args_f32", "data_i32",
                         "data_f32", "extr_i32", "extr_f32"};
  ID3D11Buffer *buffers[] = {g_args_i32_buf, g_args_f32_buf, g_data_i32_buf,
                             g_data_f32_buf, g_extr_i32_buf, g_extr_f32_buf};

  for (int i = 0; i < 6; i++) {
    char *data;
    size_t len;
    data = DumpBuffer(buffers[i], &len);
    FILE *f = fopen(names[i], "wb");
    fwrite(data, 1, len, f);
    fclose(f);
    printf("Wrote %d bytes to %s\n", static_cast<int>(len), names[i]);
    delete[] data;
  }
}

}

TLANG_NAMESPACE_END
