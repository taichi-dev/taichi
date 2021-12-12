#include "taichi/backends/dx/dx_api.h"

#include <memory>

#include "taichi/backends/dx/dx_device.h"

namespace taichi {
namespace lang {
namespace directx11 {

// The Structured Buffer created in this way will have no CPU access.
HRESULT create_structured_buffer(ID3D11Device *device,
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

HRESULT create_raw_buffer(ID3D11Device *device,
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

HRESULT create_cpu_accessible_buffer_copy(ID3D11Device *device,
                                          ID3D11Buffer *src_buf,
                                          ID3D11Buffer **out_buf) {
  D3D11_BUFFER_DESC desc;
  src_buf->GetDesc(&desc);
  D3D11_BUFFER_DESC desc1 = {};
  desc1.BindFlags = 0;
  desc1.ByteWidth = desc.ByteWidth;
  desc1.Usage = D3D11_USAGE_STAGING;
  desc1.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE | D3D11_CPU_ACCESS_READ;
  desc1.MiscFlags = 0;
  HRESULT hr = device->CreateBuffer(&desc1, nullptr, out_buf);
  return hr;
}

HRESULT create_buffer_uav(ID3D11Device *device,
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

HRESULT compile_compute_shader_from_string(const std::string &source,
                                           LPCSTR entry_point,
                                           ID3D11Device *device,
                                           ID3DBlob **blob) {
  UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL2;
  LPCSTR profile = (device->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_0)
                       ? "cs_5_0"
                       : "cs_4_0";
  const D3D_SHADER_MACRO defines[] = {"EXAMPLE_DEFINE", "1", NULL, NULL};
  ID3DBlob *shader_blob = nullptr, *error_blob = nullptr;
  HRESULT hr =
      D3DCompile(source.data(), source.size(), nullptr, defines, nullptr,
                 entry_point, profile, flags, 0, &shader_blob, &error_blob);
  if (FAILED(hr)) {
    TI_WARN("Error in compile_compute_shader_from_string\n");
    if (error_blob) {
      TI_WARN("{}", (char *)error_blob->GetBufferPointer());
      error_blob->Release();
    } else
      TI_WARN("error_blob is null\n");
    if (shader_blob) {
      shader_blob->Release();
    }
    return hr;
  }
  *blob = shader_blob;
  return hr;
}

#ifdef DX_API_CREATE_DEBUG_WINDOW
LRESULT CALLBACK WindowProc(HWND hWnd,
                            UINT message,
                            WPARAM wParam,
                            LPARAM lParam) {
  switch (message) {
    case WM_DESTROY: {
      PostQuitMessage(0);
      return 0;
    }
    default:
      break;
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}
#endif

bool is_dx_api_available() {
#ifdef TI_WITH_DX11
  DxDevice::create_dx11_device();
  return true;
#else
  return false;
#endif
}

std::unique_ptr<Device> get_dx_device() {
  auto device = std::make_unique<DxDevice>();
  device->set_cap(DeviceCapability::spirv_version, 0x10300);
  return std::move(device);
}

}  // namespace directx11
}  // namespace lang
}  // namespace taichi