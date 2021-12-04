#include <unordered_map>

#include "taichi/backends/dx/dx_device.h"
#include "taichi/backends/dx/dx_api.h"
#include "spirv_hlsl.hpp"

namespace taichi {
namespace lang {
namespace directx11 {

extern IDXGISwapChain *g_swapchain;

static uint32_t g_alloc_id = 0;

// Mapping from binding ID to Buffer
static std::unordered_map<uint32_t, ID3D11Buffer *> g_binding2buf;

// Mapping from binding ID to CPU-accessible copy of the resource
static std::unordered_map<uint32_t, ID3D11Buffer *> g_binding2cpucopy;

// Mapping from binding ID to UAV
static std::unordered_map<uint32_t, ID3D11UnorderedAccessView *> g_binding2uav;

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
  ID3D11UnorderedAccessView *uav = g_binding2uav[alloc.alloc_id];
  binding_map_[binding] = uav;
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
  rw_buffer(set, binding, alloc);
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

DxPipeline::DxPipeline(const PipelineSourceDesc &desc,
                       const std::string &name) {
  TI_ASSERT(desc.type == PipelineSourceType::hlsl_src ||
            desc.type == PipelineSourceType::spirv_binary);

  if (desc.type == PipelineSourceType::hlsl_src) {
    TI_NOT_IMPLEMENTED;
  } else {
    std::vector<uint32_t> spirv_binary(
        (uint32_t *)desc.data, (uint32_t *)((uint8_t *)desc.data + desc.size));
    spirv_cross::CompilerHLSL hlsl(std::move(spirv_binary));
    spirv_cross::CompilerHLSL::Options options;
    options.shader_model = 40;
    hlsl.set_hlsl_options(options);

    std::string source = hlsl.compile();
    TI_TRACE("hlsl source: \n{}", source);

    ID3DBlob *shader_blob;
    ID3D11Device *device = GetD3D11Device();
    HRESULT hr = CompileComputeShaderFromString(source, "main",
                                                device, &shader_blob);
    if (SUCCEEDED(hr)) {
      hr = device->CreateComputeShader(shader_blob->GetBufferPointer(),
                                       shader_blob->GetBufferSize(), nullptr,
                                       &compute_shader_);
      shader_blob->Release();
      compute_shader_->SetPrivateData(WKPDID_D3DDebugObjectName,
                                      name.size(), name.c_str());
    } else {
      TI_ERROR("HLSL compute shader compilation error");
    }
  }
}

DxPipeline::~DxPipeline() {
}

ResourceBinder *DxPipeline::resource_binder() {
  return &binder_;
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
  for (auto &[binding, uav] : binder->binding_map()) {
    std::unique_ptr<CmdBindBufferToIndex> cmd =
        std::make_unique<CmdBindBufferToIndex>();
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
  cmd->src = g_binding2buf.at(src.alloc_id);
  cmd->dst = g_binding2buf.at(dst.alloc_id);
  cmd->src_offset = src.offset;
  cmd->dst_offset = dst.offset;
  cmd->size = size;
  recorded_commands_.push_back(std::move(cmd));
}

void DxCommandList::buffer_fill(DevicePtr ptr, size_t size, uint32_t data) {
  std::unique_ptr<DxCommandList::CmdBufferFill> cmd =
      std::make_unique<CmdBufferFill>();
  ID3D11Buffer *buf = g_binding2buf.at(ptr.alloc_id);
  ID3D11UnorderedAccessView *uav = g_binding2uav.at(ptr.alloc_id);
  cmd->uav = uav;
  D3D11_BUFFER_DESC desc;
  buf->GetDesc(&desc);
  cmd->size = desc.ByteWidth;
  recorded_commands_.push_back(std::move(cmd));
}

void DxCommandList::CmdBindPipeline::execute() {
  GetD3D11Context()->CSSetShader(compute_shader_, nullptr, 0);
}

void DxCommandList::CmdBindBufferToIndex::execute() {
  char pdata[100];
  UINT data_size = 100;
  uav->GetPrivateData(WKPDID_D3DDebugObjectName, &data_size, pdata);
  GetD3D11Context()->CSSetUnorderedAccessViews(binding, 1, &uav, nullptr);
}

void DxCommandList::CmdBufferFill::execute() {
  ID3D11DeviceContext *context = GetD3D11Context();
  const UINT values[4] = {data, data, data, data};
  context->ClearUnorderedAccessViewUint(uav, values);
}

void DxCommandList::CmdBufferCopy::execute() {
  D3D11_BOX box;
  box.left = src_offset;
  box.right = src_offset + size;
  box.front = box.top = 0;
  box.back = box.bottom = 1;
  GetD3D11Context()->CopySubresourceRegion(dst, 0, dst_offset, 0, 0, src, 0, &box);
}

void DxCommandList::CmdDispatch::execute() {
  GetD3D11Context()->Dispatch(x, y, z);
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

DxStream::~DxStream() {
}

std::unique_ptr<CommandList> DxStream::new_command_list() {
  return std::make_unique<DxCommandList>();
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

DxDevice::~DxDevice() {
  if (g_swapchain) {
    g_swapchain->Present(0, 0);
  }
}

DeviceAllocation DxDevice::allocate_memory(const AllocParams &params) {
  ID3D11Buffer *buf;
  ID3D11Device *device = GetD3D11Device();

  CreateRawBuffer(device, params.size, nullptr, &buf);
  g_binding2buf[g_alloc_id] = buf;

  ID3D11UnorderedAccessView *uav;
  HRESULT hr = CreateBufferUAV(device, buf, &uav);
  if (!SUCCEEDED(hr)) {
    TI_ERROR("Could not create UAV for the view\n");
  }
  g_binding2uav[g_alloc_id] = uav;

  // set debug name
  std::string buf_name = "buffer alloc#" + std::to_string(g_alloc_id) +
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
  alloc.alloc_id = g_alloc_id;

  g_alloc_id++;
  return alloc;
}

void DxDevice::dealloc_memory(DeviceAllocation handle) {
  uint32_t alloc_id = handle.alloc_id;
  ID3D11Buffer *buf = g_binding2buf[alloc_id];
  buf->Release();

  if (g_binding2cpucopy.count(alloc_id) > 0) {
    g_binding2cpucopy[alloc_id]->Release();
  }
}

void *DxDevice::map_range(DevicePtr ptr, uint64_t size) {
  TI_NOT_IMPLEMENTED;
}

void *DxDevice::map(DeviceAllocation alloc) {
  uint32_t alloc_id = alloc.alloc_id;
  ID3D11Buffer *buf = g_binding2buf.at(alloc_id);
  ID3D11Buffer *cpucopy;

  if (g_binding2cpucopy.find(alloc_id) == g_binding2cpucopy.end()) {
    //
    // The reason we need a CPU-accessible copy is: a Resource that can be
    // bound to the GPU can't be CPU-accessible in DX11
    //
    CreateCPUAccessibleCopyOfBuffer(GetD3D11Device(), buf, &cpucopy);
    const std::string n = "CPU copy of alloc #" + std::to_string(alloc_id);
    cpucopy->SetPrivateData(WKPDID_D3DDebugObjectName, n.size(), n.c_str());
    g_binding2cpucopy[alloc_id] = cpucopy;
  } else {
    cpucopy = g_binding2cpucopy.at(alloc_id);
  }

  //
  // 
  // TODO:
  // This map and unmap is currently very slow b/c it copies the CopyResource.
  // Need a way to copy as little as possible
  // 
  // read-modify-write
  ID3D11DeviceContext *context = GetD3D11Context();
  context->CopyResource(cpucopy, buf);
  D3D11_MAPPED_SUBRESOURCE mapped;
  context->Map(cpucopy, 0, D3D11_MAP_READ_WRITE, 0, &mapped);
  return mapped.pData;
}

void DxDevice::unmap(DevicePtr ptr) {
  ID3D11Buffer *cpucopy = g_binding2cpucopy.at(ptr.alloc_id);
  ID3D11DeviceContext *context = GetD3D11Context();
  context->Unmap(cpucopy, 0);
}

void DxDevice::unmap(DeviceAllocation alloc) {
  ID3D11Buffer *cpucopy = g_binding2cpucopy.at(alloc.alloc_id);
  ID3D11Buffer *buf = g_binding2buf.at(alloc.alloc_id);
  ID3D11DeviceContext *context = GetD3D11Context();
  context->Unmap(cpucopy, 0);
  context->CopyResource(buf, cpucopy);
}
void DxDevice::memcpy_internal(DevicePtr dst,
                               DevicePtr src,
                               uint64_t size) {
  TI_NOT_IMPLEMENTED;
}

Stream *DxDevice::get_compute_stream() {
  return &stream_;
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

std::unique_ptr<Surface> DxDevice::create_surface(
    const SurfaceConfig &config) {
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

std::unique_ptr<Pipeline> DxDevice::create_pipeline(
    const PipelineSourceDesc &src,
  std::string name){
  return std::make_unique<DxPipeline>(src, name);
}

}  // namespace dx
}  // namespace lang
}  // namespace taichi