#include "taichi/backends/dx/dx_api.h"

#include <memory>

#include "taichi/backends/dx/dx_device.h"

TLANG_NAMESPACE_BEGIN

namespace dx {

ID3D11Device *g_device;
ID3D11DeviceContext *g_context;
IDXGISwapChain *g_swapchain;
ID3D11Buffer *g_args_i32_buf, *g_args_f32_buf, *g_data_i32_buf, *g_data_f32_buf,
    *g_extr_i32_buf, *g_extr_f32_buf, *g_locks_buf, *tmp_arg_buf;
ID3D11UnorderedAccessView *g_args_i32_uav, *g_args_f32_uav, *g_data_i32_uav,
    *g_data_f32_uav, *g_extr_i32_uav, *g_extr_f32_uav, *g_locks_uav;

ID3D11Device *GetD3D11Device() {
  return g_device;
}

ID3D11DeviceContext *GetD3D11Context() {
  return g_context;
}

ID3D11Buffer *GetTmpArgBuf() {
  return tmp_arg_buf;
}

HRESULT CreateComputeDevice(ID3D11Device **out_device,
                            ID3D11DeviceContext **out_context,
                            HWND hWnd,
                            IDXGISwapChain **out_swapchain,
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
  HRESULT hr;

  if (hWnd != 0 && out_swapchain != nullptr) {
    DXGI_SWAP_CHAIN_DESC scd;
    ZeroMemory(&scd, sizeof(scd));
    scd.BufferCount = 1;
    scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow = hWnd;
    scd.SampleDesc.Count = 4;
    scd.Windowed = true;
    hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, NULL, NULL, NULL,
        D3D11_SDK_VERSION, &scd, out_swapchain, &device, NULL, &context);
  } else {
    hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
                           levels, _countof(levels), D3D11_SDK_VERSION, &device,
                           nullptr, &context);
  }

  if (FAILED(hr)) {
    TI_ERROR("Failed to create D3D11 device: %08X\n", hr);
  }

  if (device->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0) {
    D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS hwopts = {0};
    device->CheckFeatureSupport(D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, &hwopts,
                                sizeof(hwopts));
    if (!hwopts.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x) {
      device->Release();
      TI_ERROR(
          "DirectCompute not supported via "
          "ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4");
    }
  }

  *out_device = device;
  *out_context = context;
  return hr;
}

// The Structured Buffer created in this way will have no CPU access.
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

HRESULT CreateCPUAccessibleCopyOfBuffer(ID3D11Device *device,
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
    TI_ERROR("Error in CompileComputeShaderFromString\n");
    if (error_blob) {
      TI_ERROR("%s\n", (char *)error_blob->GetBufferPointer());
      error_blob->Release();
    } else
      TI_ERROR("error_blob is null\n");
    if (shader_blob) {
      shader_blob->Release();
    }
    fflush(stdout);
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

int WIN_W = 320, WIN_H = 240;

bool initialize_dx(bool error_tolerance) {
  if (g_device == nullptr || g_context == nullptr) {
    TI_TRACE("Creating D3D11 device");
    HWND hWnd = 0;
    IDXGISwapChain **pp_swapchain = nullptr;

    bool is_create_window = true;

    #ifdef DX_API_CREATE_DEBUG_WINDOW
    if (is_create_window) {
      pp_swapchain = &g_swapchain;
      // stolen from win32.cpp
      int width = 320, height = 240;
      std::wstring window_name = L"Taichi DX test window";
      auto CLASS_NAME = L"Taichi Win32 Window";

      WNDCLASS wc = {};

      wc.lpfnWndProc = WindowProc;
      wc.hInstance = GetModuleHandle(0);
      wc.lpszClassName = CLASS_NAME;

      RegisterClass(&wc);

      RECT window_rect;
      window_rect.left = 0;
      window_rect.right = width;
      window_rect.top = 0;
      window_rect.bottom = height;

      AdjustWindowRect(&window_rect, WS_OVERLAPPEDWINDOW, false);

      hWnd = CreateWindowEx(0,           // Optional window styles.
                            CLASS_NAME,  // Window class
                            std::wstring(window_name.begin(), window_name.end())
                                .data(),          // Window text
                            WS_OVERLAPPEDWINDOW,  // Window style
                            // Size and position
                            CW_USEDEFAULT, CW_USEDEFAULT,
                            window_rect.right - window_rect.left,
                            window_rect.bottom - window_rect.top,
                            NULL,                // Parent window
                            NULL,                // Menu
                            GetModuleHandle(0),  // Instance handle
                            NULL                 // Additional application data
      );
      TI_ERROR_IF(hWnd == NULL, "Window creation failed");
      ShowWindow(hWnd, SW_SHOWDEFAULT);
    }
    #endif

    CreateComputeDevice(&g_device, &g_context, hWnd, pp_swapchain, false);

    if (g_swapchain) {
      g_swapchain->Present(0, 0);
      g_swapchain->Present(0, 0);
    }

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
    tmp_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
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

std::unique_ptr<Device> get_dx_device() {
  auto device = std::make_unique<DxDevice>();
  device->set_cap(DeviceCapability::spirv_version, 0x10300);
  return std::move(device);
}

}

TLANG_NAMESPACE_END