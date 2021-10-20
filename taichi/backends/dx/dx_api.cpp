#include "taichi/backends/dx/dx_api.h"

TLANG_NAMESPACE_BEGIN

namespace dx {

ID3D11Device *g_device;
ID3D11DeviceContext *g_context;
ID3D11Buffer *g_args_i32_buf, *g_args_f32_buf, *g_data_i32_buf, *g_data_f32_buf,
    *g_extr_i32_buf, *g_extr_f32_buf, *g_locks_buf, *tmp_arg_buf;
ID3D11UnorderedAccessView *g_args_i32_uav, *g_args_f32_uav, *g_data_i32_uav,
    *g_data_f32_uav, *g_extr_i32_uav, *g_extr_f32_uav, *g_locks_uav;

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
  // flags |= D3D11_CREATE_DEVICE_DEBUG;

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

bool initialize_dx(bool error_tolerance) {
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
  return initialize_dx();
}

}

TLANG_NAMESPACE_END