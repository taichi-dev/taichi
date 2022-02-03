#include "taichi/backends/dx/dx_device.h"

namespace taichi {
namespace lang {
namespace directx11 {

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

Dx11CommandList::Dx11CommandList(Dx11Device *ti_device) : device_(ti_device) {
}

Dx11CommandList::~Dx11CommandList() {
}

void Dx11CommandList::bind_pipeline(Pipeline *p) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::bind_resources(ResourceBinder *binder) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::bind_resources(ResourceBinder *binder,
                                     ResourceBinder::Bindings *bindings) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::buffer_barrier(DevicePtr ptr, size_t size) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::buffer_barrier(DeviceAllocation alloc) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::memory_barrier() {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::buffer_copy(DevicePtr dst, DevicePtr src, size_t size) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::buffer_fill(DevicePtr ptr, size_t size, uint32_t data) {
  std::unique_ptr<Dx11CommandList::CmdBufferFill> cmd =
      std::make_unique<CmdBufferFill>(this);
  ID3D11Buffer *buf = device_->alloc_id_to_buffer(ptr.alloc_id);
  ID3D11UnorderedAccessView *uav = device_->alloc_id_to_uav(ptr.alloc_id);
  cmd->uav = uav;
  D3D11_BUFFER_DESC desc;
  buf->GetDesc(&desc);
  cmd->size = desc.ByteWidth;
  recorded_commands_.push_back(std::move(cmd));
}

void Dx11CommandList::CmdBufferFill::execute() {
  ID3D11DeviceContext *context = cmdlist_->device_->d3d11_context();
  const UINT values[4] = {data, data, data, data};
  context->ClearUnorderedAccessViewUint(uav, values);
}

void Dx11CommandList::dispatch(uint32_t x, uint32_t y, uint32_t z) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::begin_renderpass(int x0,
                                       int y0,
                                       int x1,
                                       int y1,
                                       uint32_t num_color_attachments,
                                       DeviceAllocation *color_attachments,
                                       bool *color_clear,
                                       std::vector<float> *clear_colors,
                                       DeviceAllocation *depth_attachment,
                                       bool depth_clear) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::end_renderpass() {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::draw(uint32_t num_verticies, uint32_t start_vertex) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::clear_color(float r, float g, float b, float a) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::set_line_width(float width) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::draw_indexed(uint32_t num_indicies,
                                   uint32_t start_vertex,
                                   uint32_t start_index) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::image_transition(DeviceAllocation img,
                                       ImageLayout old_layout,
                                       ImageLayout new_layout) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::buffer_to_image(DeviceAllocation dst_img,
                                      DevicePtr src_buf,
                                      ImageLayout img_layout,
                                      const BufferImageCopyParams &params) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::image_to_buffer(DevicePtr dst_buf,
                                      DeviceAllocation src_img,
                                      ImageLayout img_layout,
                                      const BufferImageCopyParams &params) {
  TI_NOT_IMPLEMENTED;
}

void Dx11CommandList::run_commands() {
  for (const auto &cmd : recorded_commands_) {
    cmd->execute();
  }
}

namespace {
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

}  // namespace

Dx11Device::Dx11Device() {
  create_dx11_device();
  if (kD3d11DebugEnabled) {
    info_queue_ = std::make_unique<Dx11InfoQueue>(device_);
  }
  set_cap(DeviceCapability::spirv_version, 0x10300);

  stream_ = new Dx11Stream(this);
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
  ID3D11Buffer *buf;
  HRESULT hr;
  hr = create_raw_buffer(device_, params.size, nullptr, &buf);
  check_dx_error(hr, "create raw buffer");
  alloc_id_to_buffer_[alloc_serial_] = buf;

  ID3D11UnorderedAccessView *uav;
  hr = create_buffer_uav(device_, buf, &uav);
  check_dx_error(hr, "create UAV for buffer");
  alloc_id_to_uav_[alloc_serial_] = uav;

  // Set debug names
  std::string buf_name = "buffer alloc#" + std::to_string(alloc_serial_) +
                         " size=" + std::to_string(params.size) + '\0';
  hr = buf->SetPrivateData(WKPDID_D3DDebugObjectName, buf_name.size(),
                           buf_name.c_str());
  check_dx_error(hr, "set name for buffer");

  std::string uav_name = "UAV of " + buf_name;
  hr = uav->SetPrivateData(WKPDID_D3DDebugObjectName, uav_name.size(),
                           uav_name.c_str());
  check_dx_error(hr, "set name for UAV");

  DeviceAllocation alloc;
  alloc.device = this;
  alloc.alloc_id = alloc_serial_;
  ++alloc_serial_;

  return alloc;
}

void Dx11Device::dealloc_memory(DeviceAllocation handle) {
  uint32_t alloc_id = handle.alloc_id;
  ID3D11Buffer *buf = alloc_id_to_buffer_[alloc_id];
  buf->Release();
  alloc_id_to_buffer_.erase(alloc_id);
  ID3D11UnorderedAccessView *uav = alloc_id_to_uav_[alloc_id];
  uav->Release();
  alloc_id_to_uav_.erase(alloc_id);
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
  return stream_;
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

ID3D11Buffer *Dx11Device::alloc_id_to_buffer(uint32_t alloc_id) {
  return alloc_id_to_buffer_.at(alloc_id);
}

ID3D11Buffer *Dx11Device::alloc_id_to_buffer_cpu_copy(uint32_t alloc_id) {
  return alloc_id_to_cpucopy_.at(alloc_id);
}

ID3D11UnorderedAccessView *Dx11Device::alloc_id_to_uav(uint32_t alloc_id) {
  return alloc_id_to_uav_.at(alloc_id);
}

Dx11Stream::Dx11Stream(Dx11Device *device_) : device_(device_) {
}

Dx11Stream::~Dx11Stream() {
}

std::unique_ptr<CommandList> Dx11Stream::new_command_list() {
  return std::make_unique<Dx11CommandList>(device_);
}

void Dx11Stream::submit(CommandList *cmdlist) {
  TI_NOT_IMPLEMENTED;
}

// No difference for DX11
void Dx11Stream::submit_synced(CommandList *cmdlist) {
  Dx11CommandList *dx_cmd_list = static_cast<Dx11CommandList *>(cmdlist);
  dx_cmd_list->run_commands();
}

void Dx11Stream::command_sync() {
  // Not needed for DX11
}

}  // namespace directx11
}  // namespace lang
}  // namespace taichi
