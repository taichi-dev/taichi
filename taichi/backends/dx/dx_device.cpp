#include <unordered_map>
#include <memory>

#include "taichi/backends/dx/dx_device.h"
#include "taichi/backends/dx/dx_api.h"
#include "spirv_hlsl.hpp"

namespace taichi {
namespace lang {
namespace directx11 {

ID3D11Device *DxDevice::device_;
ID3D11DeviceContext *DxDevice::context_;

DxResourceBinder::~DxResourceBinder() {
}

std::unique_ptr<ResourceBinder::Bindings> DxResourceBinder::materialize() {
  TI_NOT_IMPLEMENTED;
  return nullptr;
}

void DxResourceBinder::rw_buffer(uint32_t set,
                                 uint32_t binding,
                                 DevicePtr ptr,
                                 size_t size) {
  TI_NOT_IMPLEMENTED;
}

void DxResourceBinder::rw_buffer(uint32_t set,
                                 uint32_t binding,
                                 DeviceAllocation alloc) {
  TI_ASSERT_INFO(set == 0, "DX only supports set = 0, requested set = {}", set);
  binding_to_alloc_id_[binding] = alloc.alloc_id;
}

void DxResourceBinder::buffer(uint32_t set,
                              uint32_t binding,
                              DevicePtr ptr,
                              size_t size) {
  TI_NOT_IMPLEMENTED;
}

void DxResourceBinder::buffer(uint32_t set,
                              uint32_t binding,
                              DeviceAllocation alloc) {
  // TODO: handle this with constant buffer handling
  TI_NOT_IMPLEMENTED;
}

void DxResourceBinder::image(uint32_t set,
                             uint32_t binding,
                             DeviceAllocation alloc,
                             ImageSamplerConfig sampler_config) {
  TI_NOT_IMPLEMENTED;
}

// Set vertex buffer (not implemented in compute only device)
void DxResourceBinder::vertex_buffer(DevicePtr ptr, uint32_t binding) {
  TI_NOT_IMPLEMENTED;
}

// Set index buffer (not implemented in compute only device)
// index_width = 4 -> uint32 index
// index_width = 2 -> uint16 index
void DxResourceBinder::index_buffer(DevicePtr ptr, size_t index_width) {
  TI_NOT_IMPLEMENTED;
}

DxPipeline::DxPipeline(const PipelineSourceDesc &desc, const std::string &name)
    : compute_shader_(nullptr) {
  // TODO: Currently, PipelineSourceType::hlsl_src still returns SPIRV binary.
  // Will need to update this section when that changes
  TI_ASSERT(desc.type == PipelineSourceType::hlsl_src ||
            desc.type == PipelineSourceType::spirv_binary);

  ID3D11Device *device = DxDevice::device_;
  ID3DBlob *shader_blob;
  HRESULT hr;

  std::vector<uint32_t> spirv_binary(
      (uint32_t *)desc.data, (uint32_t *)((uint8_t *)desc.data + desc.size));
  spirv_cross::CompilerHLSL hlsl(std::move(spirv_binary));
  spirv_cross::CompilerHLSL::Options options;
  options.shader_model = 40;
  hlsl.set_hlsl_options(options);

  std::string source = hlsl.compile();
  TI_TRACE("hlsl source: \n{}", source);

  hr = compile_compute_shader_from_string(source, "main", device, &shader_blob);
  if (SUCCEEDED(hr)) {
    hr = device->CreateComputeShader(shader_blob->GetBufferPointer(),
                                     shader_blob->GetBufferSize(), nullptr,
                                     &compute_shader_);
    shader_blob->Release();
    compute_shader_->SetPrivateData(WKPDID_D3DDebugObjectName, name.size(),
                                    name.c_str());
    if (!SUCCEEDED(hr)) {
      TI_ERROR("HLSL compute shader creation error");
    }
  } else {
    TI_ERROR("HLSL compute shader compilation error");
  }
}

DxPipeline::~DxPipeline() {
}

ResourceBinder *DxPipeline::resource_binder() {
  return &binder_;
}

DxCommandList::DxCommandList(DxDevice *ti_device) : device_(ti_device) {
}

DxCommandList::~DxCommandList() {
}

void DxCommandList::bind_pipeline(Pipeline *p) {
  DxPipeline *pipeline = static_cast<DxPipeline *>(p);
  std::unique_ptr<CmdBindPipeline> cmd = std::make_unique<CmdBindPipeline>();
  cmd->compute_shader_ = pipeline->get_program();
  recorded_commands_.push_back(std::move(cmd));
}

void DxCommandList::bind_resources(ResourceBinder *binder_) {
  DxResourceBinder *binder = static_cast<DxResourceBinder *>(binder_);
  for (auto &[binding, alloc_id] : binder->binding_to_alloc_id()) {
    std::unique_ptr<CmdBindBufferToIndex> cmd =
        std::make_unique<CmdBindBufferToIndex>();
    ID3D11UnorderedAccessView *uav = device_->alloc_id_to_uav(alloc_id);
    cmd->binding = binding;
    cmd->uav = uav;
    recorded_commands_.push_back(std::move(cmd));
  }
}

void DxCommandList::bind_resources(ResourceBinder *binder,
                                   ResourceBinder::Bindings *bindings) {
  TI_NOT_IMPLEMENTED;
}

void DxCommandList::buffer_barrier(DevicePtr ptr, size_t size) {
  TI_NOT_IMPLEMENTED;
}

void DxCommandList::buffer_barrier(DeviceAllocation alloc) {
  // Not needed for DX 11
}

void DxCommandList::memory_barrier() {
  // Not needed for DX 11
}

void DxCommandList::buffer_copy(DevicePtr dst, DevicePtr src, size_t size) {
  std::unique_ptr<CmdBufferCopy> cmd = std::make_unique<CmdBufferCopy>();
  cmd->src = device_->alloc_id_to_buffer(src.alloc_id);
  cmd->dst = device_->alloc_id_to_buffer(dst.alloc_id);
  cmd->src_offset = src.offset;
  cmd->dst_offset = dst.offset;
  cmd->size = size;
  recorded_commands_.push_back(std::move(cmd));
}

void DxCommandList::buffer_fill(DevicePtr ptr, size_t size, uint32_t data) {
  std::unique_ptr<DxCommandList::CmdBufferFill> cmd =
      std::make_unique<CmdBufferFill>();
  ID3D11Buffer *buf = device_->alloc_id_to_buffer(ptr.alloc_id);
  ID3D11UnorderedAccessView *uav = device_->alloc_id_to_uav(ptr.alloc_id);
  cmd->uav = uav;
  D3D11_BUFFER_DESC desc;
  buf->GetDesc(&desc);
  cmd->size = desc.ByteWidth;
  recorded_commands_.push_back(std::move(cmd));
}

void DxCommandList::CmdBindPipeline::execute() {
  DxDevice::context_->CSSetShader(compute_shader_, nullptr, 0);
}

void DxCommandList::CmdBindBufferToIndex::execute() {
  char pdata[100];
  UINT data_size = 100;
  uav->GetPrivateData(WKPDID_D3DDebugObjectName, &data_size, pdata);
  DxDevice::context_->CSSetUnorderedAccessViews(binding, 1, &uav, nullptr);
}

void DxCommandList::CmdBufferFill::execute() {
  ID3D11DeviceContext *context = DxDevice::context_;
  const UINT values[4] = {data, data, data, data};
  context->ClearUnorderedAccessViewUint(uav, values);
}

void DxCommandList::CmdBufferCopy::execute() {
  D3D11_BOX box;
  box.left = src_offset;
  box.right = src_offset + size;
  box.front = box.top = 0;
  box.back = box.bottom = 1;
  DxDevice::context_->CopySubresourceRegion(dst, 0, dst_offset, 0, 0, src, 0,
                                            &box);
}

void DxCommandList::CmdDispatch::execute() {
  DxDevice::context_->Dispatch(x, y, z);
}

void DxCommandList::dispatch(uint32_t x, uint32_t y, uint32_t z) {
  std::unique_ptr<CmdDispatch> cmd = std::make_unique<CmdDispatch>();
  cmd->x = x;
  cmd->y = y;
  cmd->z = z;
  recorded_commands_.push_back(std::move(cmd));
}

// These are not implemented in compute only device
void DxCommandList::begin_renderpass(int x0,
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

void DxCommandList::end_renderpass() {
  TI_NOT_IMPLEMENTED;
}

void DxCommandList::draw(uint32_t num_verticies, uint32_t start_vertex) {
  TI_NOT_IMPLEMENTED;
}

void DxCommandList::clear_color(float r, float g, float b, float a) {
  TI_NOT_IMPLEMENTED;
}

void DxCommandList::set_line_width(float width) {
  TI_NOT_IMPLEMENTED;
}

void DxCommandList::draw_indexed(uint32_t num_indicies,
                                 uint32_t start_vertex,
                                 uint32_t start_index) {
  TI_NOT_IMPLEMENTED;
}

void DxCommandList::image_transition(DeviceAllocation img,
                                     ImageLayout old_layout,
                                     ImageLayout new_layout) {
  TI_NOT_IMPLEMENTED;
}

void DxCommandList::buffer_to_image(DeviceAllocation dst_img,
                                    DevicePtr src_buf,
                                    ImageLayout img_layout,
                                    const BufferImageCopyParams &params) {
  TI_NOT_IMPLEMENTED;
}

void DxCommandList::image_to_buffer(DevicePtr dst_buf,
                                    DeviceAllocation src_img,
                                    ImageLayout img_layout,
                                    const BufferImageCopyParams &params) {
  TI_NOT_IMPLEMENTED;
}

void DxCommandList::run_commands() {
  for (const auto &cmd : recorded_commands_) {
    cmd->execute();
  }
}

DxStream::DxStream(DxDevice *device_) : device_(device_) {
}

DxStream::~DxStream() {
}

std::unique_ptr<CommandList> DxStream::new_command_list() {
  return std::make_unique<DxCommandList>(device_);
}

void DxStream::submit(CommandList *cmdlist) {
  DxCommandList *dx_cmd_list = static_cast<DxCommandList *>(cmdlist);
  dx_cmd_list->run_commands();
}

// No difference for DX11
void DxStream::submit_synced(CommandList *cmdlist) {
  DxCommandList *dx_cmd_list = static_cast<DxCommandList *>(cmdlist);
  dx_cmd_list->run_commands();
}

void DxStream::command_sync() {
  // Not needed for DX11
}

DxDevice::DxDevice() : alloc_serial_(0) {
  stream_ = new DxStream(this);
  set_cap(DeviceCapability::spirv_version, 0x10300);
}

DxDevice::~DxDevice() {
}

HRESULT create_compute_device(ID3D11Device **out_device,
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

void DxDevice::create_dx11_device() {
  if (device_ == nullptr || context_ == nullptr) {
    TI_TRACE("Creating D3D11 device");
    HWND hWnd = 0;
    IDXGISwapChain **pp_swapchain = nullptr;

#ifdef DX_API_CREATE_DEBUG_WINDOW
    bool is_create_window = true;
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

    if (g_swapchain) {
      g_swapchain->Present(0, 0);
      g_swapchain->Present(0, 0);
    }

#endif
    create_compute_device(&device_, &context_, hWnd, pp_swapchain, false);
  } else {
    TI_TRACE("D3D11 device has already been created.");
  }
}

DeviceAllocation DxDevice::allocate_memory(const AllocParams &params) {
  ID3D11Buffer *buf;
  ID3D11Device *device = DxDevice::device_;

  create_raw_buffer(device, params.size, nullptr, &buf);
  alloc_id_to_buffer_[alloc_serial_] = buf;

  ID3D11UnorderedAccessView *uav;
  HRESULT hr = create_buffer_uav(device, buf, &uav);
  if (!SUCCEEDED(hr)) {
    TI_ERROR("Could not create UAV for the view\n");
  }
  alloc_id_to_uav_[alloc_serial_] = uav;

  // set debug name
  std::string buf_name = "buffer alloc#" + std::to_string(alloc_serial_) +
                         " size=" + std::to_string(params.size) + '\0';
  hr = buf->SetPrivateData(WKPDID_D3DDebugObjectName, buf_name.size(),
                           buf_name.c_str());
  assert(SUCCEEDED(hr));
  std::string uav_name = "UAV of " + buf_name;
  hr = uav->SetPrivateData(WKPDID_D3DDebugObjectName, uav_name.size(),
                           uav_name.c_str());
  assert(SUCCEEDED(hr));

  DeviceAllocation alloc;
  alloc.device = this;
  alloc.alloc_id = alloc_serial_;

  alloc_serial_++;
  return alloc;
}

void DxDevice::dealloc_memory(DeviceAllocation handle) {
  uint32_t alloc_id = handle.alloc_id;
  ID3D11Buffer *buf = alloc_id_to_buffer_[alloc_id];
  buf->Release();

  if (alloc_id_to_buffer_.count(alloc_id) > 0) {
    alloc_id_to_buffer_[alloc_id]->Release();
    alloc_id_to_buffer_.erase(alloc_id);
  }
}

void *DxDevice::map_range(DevicePtr ptr, uint64_t size) {
  TI_NOT_IMPLEMENTED;
}

void *DxDevice::map(DeviceAllocation alloc) {
  uint32_t alloc_id = alloc.alloc_id;
  ID3D11Buffer *buf = alloc_id_to_buffer_.at(alloc_id);
  ID3D11Buffer *cpucopy;

  if (alloc_id_to_cpucopy_.find(alloc_id) == alloc_id_to_cpucopy_.end()) {
    //
    // The reason we need a CPU-accessible copy is: a Resource that can be
    // bound to the GPU can't be CPU-accessible in DX11
    //
    create_cpu_accessible_buffer_copy(DxDevice::device_, buf, &cpucopy);
    const std::string n = "CPU copy of alloc #" + std::to_string(alloc_id);
    cpucopy->SetPrivateData(WKPDID_D3DDebugObjectName, n.size(), n.c_str());
    alloc_id_to_cpucopy_[alloc_id] = cpucopy;
  } else {
    cpucopy = alloc_id_to_cpucopy_.at(alloc_id);
  }

  //
  //
  // TODO:
  // This map and unmap is currently very slow b/c it copies the CopyResource.
  // Need a way to copy as little as possible
  //
  // read-modify-write
  ID3D11DeviceContext *context = DxDevice::context_;
  context->CopyResource(cpucopy, buf);
  D3D11_MAPPED_SUBRESOURCE mapped;
  context->Map(cpucopy, 0, D3D11_MAP_READ_WRITE, 0, &mapped);
  return mapped.pData;
}

void DxDevice::unmap(DevicePtr ptr) {
  ID3D11Buffer *cpucopy = alloc_id_to_cpucopy_.at(ptr.alloc_id);
  ID3D11DeviceContext *context = DxDevice::context_;
  context->Unmap(cpucopy, 0);
}

void DxDevice::unmap(DeviceAllocation alloc) {
  ID3D11Buffer *cpucopy = alloc_id_to_cpucopy_.at(alloc.alloc_id);
  ID3D11Buffer *buf = alloc_id_to_buffer_.at(alloc.alloc_id);
  ID3D11DeviceContext *context = DxDevice::context_;
  context->Unmap(cpucopy, 0);
  context->CopyResource(buf, cpucopy);
}
void DxDevice::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {
  TI_NOT_IMPLEMENTED;
}

Stream *DxDevice::get_compute_stream() {
  return stream_;
}

std::unique_ptr<Pipeline> DxDevice::create_raster_pipeline(
    const std::vector<PipelineSourceDesc> &src,
    const RasterParams &raster_params,
    const std::vector<VertexInputBinding> &vertex_inputs,
    const std::vector<VertexInputAttribute> &vertex_attrs,
    std::string name) {
  TI_NOT_IMPLEMENTED;
}

Stream *DxDevice::get_graphics_stream() {
  TI_NOT_IMPLEMENTED;
}

std::unique_ptr<Surface> DxDevice::create_surface(const SurfaceConfig &config) {
  TI_NOT_IMPLEMENTED;
}

DeviceAllocation DxDevice::create_image(const ImageParams &params) {
  TI_NOT_IMPLEMENTED;
}
void DxDevice::destroy_image(DeviceAllocation handle) {
  TI_NOT_IMPLEMENTED;
}

void DxDevice::image_transition(DeviceAllocation img,
                                ImageLayout old_layout,
                                ImageLayout new_layout) {
  TI_NOT_IMPLEMENTED;
}

void DxDevice::buffer_to_image(DeviceAllocation dst_img,
                               DevicePtr src_buf,
                               ImageLayout img_layout,
                               const BufferImageCopyParams &params) {
  TI_NOT_IMPLEMENTED;
}

void DxDevice::image_to_buffer(DevicePtr dst_buf,
                               DeviceAllocation src_img,
                               ImageLayout img_layout,
                               const BufferImageCopyParams &params) {
  TI_NOT_IMPLEMENTED;
}

ID3D11Buffer *DxDevice::alloc_id_to_buffer(uint32_t alloc_id) {
  return alloc_id_to_buffer_.at(alloc_id);
}

ID3D11Buffer *DxDevice::alloc_id_to_buffer_cpu_copy(uint32_t alloc_id) {
  return alloc_id_to_cpucopy_.at(alloc_id);
}

ID3D11UnorderedAccessView *DxDevice::alloc_id_to_uav(uint32_t alloc_id) {
  return alloc_id_to_uav_.at(alloc_id);
}

std::unique_ptr<Pipeline> DxDevice::create_pipeline(
    const PipelineSourceDesc &src,
    std::string name) {
  return std::make_unique<DxPipeline>(src, name);
}

}  // namespace directx11
}  // namespace lang
}  // namespace taichi