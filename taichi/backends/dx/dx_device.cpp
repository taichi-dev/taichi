#include "taichi/backends/dx/dx_device.h"

namespace taichi {
namespace lang {
namespace directx11 {

bool kD3d11DebugEnabled = false;  // D3D11 debugging is enabled. For testing.
bool kD3d11ForceRef = false;      // Force REF device. May be used to
                                  // force software rendering.

void debug_enabled(bool enabled) {
  kD3d11DebugEnabled = enabled;
}

void force_ref(bool force) {
  kD3d11ForceRef = force;
}

void check_dx_error(HRESULT hr, const char *msg) {
  if (!SUCCEEDED(hr)) {
    TI_ERROR("Error in {}: {}", msg, hr);
  }
}

Dx11ResourceBinder::~Dx11ResourceBinder() {
}

Dx11Pipeline::Dx11Pipeline(const PipelineSourceDesc &desc,
                           const std::string &name) {
  TI_NOT_IMPLEMENTED;
}

Dx11Pipeline::~Dx11Pipeline() {
}

ResourceBinder *Dx11Pipeline::resource_binder() {
  return nullptr;
}

HRESULT create_compute_device(ID3D11Device **out_device,
                              ID3D11DeviceContext **out_context,
                              bool force_ref,
                              bool debug_enabled) {
  const D3D_FEATURE_LEVEL levels[] = {
      D3D_FEATURE_LEVEL_11_1,
      D3D_FEATURE_LEVEL_11_0,
      D3D_FEATURE_LEVEL_10_1,
      D3D_FEATURE_LEVEL_10_0,
  };

  UINT flags = 0;
  if (debug_enabled)
    flags |= D3D11_CREATE_DEVICE_DEBUG;

  ID3D11Device *device = nullptr;
  ID3D11DeviceContext *context = nullptr;
  HRESULT hr;

  D3D_DRIVER_TYPE driver_types[] = {
      D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_SOFTWARE,
      D3D_DRIVER_TYPE_REFERENCE, D3D_DRIVER_TYPE_WARP};
  const char *driver_type_names[] = {
      "D3D_DRIVER_TYPE_HARDWARE", "D3D_DRIVER_TYPE_SOFTWARE",
      "D3D_DRIVER_TYPE_REFERENCE", "D3D_DRIVER_TYPE_WARP"};

  const int num_types = sizeof(driver_types) / sizeof(driver_types[0]);

  int attempt_idx = 0;
  if (force_ref) {
    attempt_idx = 2;
  }

  for (; attempt_idx < num_types; attempt_idx++) {
    D3D_DRIVER_TYPE driver_type = driver_types[attempt_idx];
    hr = D3D11CreateDevice(nullptr, driver_type, nullptr, flags, levels,
                           _countof(levels), D3D11_SDK_VERSION, &device,
                           nullptr, &context);

    if (FAILED(hr) || device == nullptr) {
      TI_WARN("Failed to create D3D11 device with type {}: {}\n", driver_type,
              driver_type_names[attempt_idx]);
      continue;
    }

    if (device->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0) {
      D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS hwopts = {0};
      device->CheckFeatureSupport(D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS,
                                  &hwopts, sizeof(hwopts));
      if (!hwopts.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x) {
        device->Release();
        TI_WARN(
            "DirectCompute not supported via "
            "ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4");
      }
      continue;
    }

    TI_INFO("Successfully created DX11 device with type {}",
            driver_type_names[attempt_idx]);
    *out_device = device;
    *out_context = context;
    break;
  }

  if (*out_device == nullptr || *out_context == nullptr) {
    TI_ERROR("Failed to create DX11 device using all {} driver types",
             num_types);
  }

  return hr;
}

Dx11Device::Dx11Device() {
  create_dx11_device();
  if (kD3d11DebugEnabled) {
    info_queue_ = std::make_unique<Dx11InfoQueue>(device_);
  }
  set_cap(DeviceCapability::spirv_version, 0x10300);
}

Dx11Device::~Dx11Device() {
  destroy_dx11_device();
}

void Dx11Device::create_dx11_device() {
  if (device_ != nullptr && context_ != nullptr) {
    TI_TRACE("D3D11 device has already been created.");
    return;
  }
  TI_TRACE("Creating D3D11 device");
  create_compute_device(&device_, &context_, kD3d11ForceRef,
                        kD3d11DebugEnabled);
}

void Dx11Device::destroy_dx11_device() {
  if (device_ != nullptr) {
    device_->Release();
    device_ = nullptr;
  }
  if (context_ != nullptr) {
    context_->Release();
    context_ = nullptr;
  }
}

int Dx11Device::live_dx11_object_count() {
  TI_ASSERT(info_queue_ != nullptr);
  return info_queue_->live_object_count();
}

DeviceAllocation Dx11Device::allocate_memory(const AllocParams &params) {
  TI_NOT_IMPLEMENTED;
}

void Dx11Device::dealloc_memory(DeviceAllocation handle) {
  TI_NOT_IMPLEMENTED;
}

std::unique_ptr<Pipeline> Dx11Device::create_pipeline(
    const PipelineSourceDesc &src,
    std::string name) {
  TI_NOT_IMPLEMENTED;
}

void *Dx11Device::map_range(DevicePtr ptr, uint64_t size) {
  TI_NOT_IMPLEMENTED;
}

void *Dx11Device::map(DeviceAllocation alloc) {
  TI_NOT_IMPLEMENTED;
}

void Dx11Device::unmap(DevicePtr ptr) {
  TI_NOT_IMPLEMENTED;
}

void Dx11Device::unmap(DeviceAllocation alloc) {
  TI_NOT_IMPLEMENTED;
}

void Dx11Device::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {
  TI_NOT_IMPLEMENTED;
}

Stream *Dx11Device::get_compute_stream() {
  TI_NOT_IMPLEMENTED;
}

std::unique_ptr<Pipeline> Dx11Device::create_raster_pipeline(
    const std::vector<PipelineSourceDesc> &src,
    const RasterParams &raster_params,
    const std::vector<VertexInputBinding> &vertex_inputs,
    const std::vector<VertexInputAttribute> &vertex_attrs,
    std::string name) {
  TI_NOT_IMPLEMENTED;
}

Stream *Dx11Device::get_graphics_stream() {
  TI_NOT_IMPLEMENTED;
}

std::unique_ptr<Surface> Dx11Device::create_surface(
    const SurfaceConfig &config) {
  TI_NOT_IMPLEMENTED;
}

DeviceAllocation Dx11Device::create_image(const ImageParams &params) {
  TI_NOT_IMPLEMENTED;
}

void Dx11Device::destroy_image(DeviceAllocation handle) {
  TI_NOT_IMPLEMENTED;
}

void Dx11Device::image_transition(DeviceAllocation img,
                                  ImageLayout old_layout,
                                  ImageLayout new_layout) {
  TI_NOT_IMPLEMENTED;
}

void Dx11Device::buffer_to_image(DeviceAllocation dst_img,
                                 DevicePtr src_buf,
                                 ImageLayout img_layout,
                                 const BufferImageCopyParams &params) {
  TI_NOT_IMPLEMENTED;
}
void Dx11Device::image_to_buffer(DevicePtr dst_buf,
                                 DeviceAllocation src_img,
                                 ImageLayout img_layout,
                                 const BufferImageCopyParams &params) {
  TI_NOT_IMPLEMENTED;
}

}  // namespace directx11
}  // namespace lang
}  // namespace taichi
