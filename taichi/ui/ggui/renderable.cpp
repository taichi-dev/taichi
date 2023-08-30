#include "taichi/ui/ggui/renderable.h"

#include "taichi/program/program.h"
#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void Renderable::init(const RenderableConfig &config, AppContext *app_context) {
  config_ = config;
  app_context_ = app_context;

  resource_set_ = app_context_->device().create_resource_set_unique();
}

void Renderable::create_buffer_with_staging(Device &device,
                                            size_t size,
                                            AllocUsage usage,
                                            DeviceAllocationUnique &buffer,
                                            DeviceAllocationUnique &staging) {
  {
    Device::AllocParams params{size, false, false, false, usage};
    auto [buf, res] = device.allocate_memory_unique(params);
    TI_ASSERT(res == RhiResult::success);
    buffer = std::move(buf);
  }

  {
    Device::AllocParams staging_params{size, true, false, false,
                                       AllocUsage::None};
    auto [buf, res] = device.allocate_memory_unique(staging_params);
    TI_ASSERT(res == RhiResult::success);
    staging = std::move(buf);
  }
}

void Renderable::init_buffers() {
  vertex_buffer_.reset();
  staging_vertex_buffer_.reset();
  index_buffer_.reset();
  staging_index_buffer_.reset();
  uniform_buffer_renderable_.reset();
  // Vertex buffers
  create_buffer_with_staging(app_context_->device(),
                             sizeof(Vertex) * max_vertices_count,
                             AllocUsage::Storage | AllocUsage::Vertex,
                             vertex_buffer_, staging_vertex_buffer_);
  // Index buffers
  if (max_indices_count) {
    create_buffer_with_staging(app_context_->device(),
                               sizeof(int) * max_indices_count,
                               AllocUsage::Storage | AllocUsage::Index,
                               index_buffer_, staging_index_buffer_);
  }

  // Uniform buffer
  if (config_.ubo_size) {
    auto [buf, res] = app_context_->device().allocate_memory_unique(
        {config_.ubo_size, /*host_write=*/true, /*host_read=*/false,
         /*export_sharing=*/false, AllocUsage::Uniform});
    TI_ASSERT(res == RhiResult::success);
    uniform_buffer_renderable_ = std::move(buf);
  }
}

void Renderable::copy_helper(Program *prog,
                             DevicePtr dst,
                             DevicePtr src,
                             DevicePtr staging,
                             size_t size) {
  if (src.device == nullptr) {
    // src is a host mapped pointer
    Device *target_device = dst.device;

    // Map the staging buffer and perform memcpy
    void *dst_ptr{nullptr};
    TI_ASSERT(target_device->map_range(staging, size, &dst_ptr) ==
              RhiResult::success);
    void *src_ptr = reinterpret_cast<uint8_t *>(src.alloc_id);
    memcpy(dst_ptr, src_ptr, size);
    target_device->unmap(staging);

    // Use device transfer stream to transfer from staing to GPU local memory
    auto stream = target_device->get_compute_stream();
    auto [cmd_list, res] = stream->new_command_list_unique();
    TI_ASSERT(res == RhiResult::success);
    cmd_list->buffer_copy(dst, staging, size);
    stream->submit_synced(cmd_list.get());
  } else if (prog && dst.device == src.device &&
             dst.device == prog->get_graphics_device()) {
    // src and dst are from the same device as GGUI
    prog->enqueue_compute_op_lambda(
        [=](Device *device, CommandList *cmdlist) {
          cmdlist->buffer_barrier(src);
          cmdlist->buffer_copy(dst, src, size);
          cmdlist->buffer_barrier(dst);
        },
        {});
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

void Renderable::update_data(const RenderableInfo &info) {
  // We might not have a current program if GGUI is used in external apps to
  // load AOT modules
  Program *prog = app_context_->prog();
  if (prog && prog->get_graphics_device() != &app_context_->device()) {
    prog->flush();
  }

  // Check if we need to update Graphics Pipeline
  if (!pipeline_ || info.display_mode != config_.polygon_mode) {
    config_.polygon_mode = info.display_mode;
    create_graphics_pipeline();
  }

  // Validate vertex buffer attributes
  static_assert(sizeof(Vertex) % sizeof(float) == 0);
  constexpr size_t vertex_size = sizeof(Vertex) / sizeof(float);
  TI_ASSERT_INFO(info.vbo.valid, "Vertex buffer must be valid");
  TI_ASSERT_INFO(info.vbo.dtype == PrimitiveType::f32,
                 "Vertex buffer must be f32");
  TI_ASSERT_INFO(
      info.vbo.num_elements % vertex_size == 0,
      "Vertex buffer size ({}) must be a multiple of Vertex struct size ({})",
      info.vbo.num_elements, vertex_size);
  const size_t vbo_size_bytes = info.vbo.num_elements * sizeof(float);
  const size_t num_vertices = info.vbo.num_elements / vertex_size;
  const size_t draw_num_vertices = info.draw_vertex_count;
  const size_t draw_first_vertices = info.draw_first_vertex % num_vertices;
  // Validate index buffer attributes
  size_t num_indices = 0;
  size_t draw_num_indices = 0;
  size_t draw_first_indices = 0;
  size_t ibo_size_bytes = 0;
  if (info.indices.valid) {
    if (info.indices.dtype != PrimitiveType::i32 &&
        info.indices.dtype != PrimitiveType::u32) {
      throw std::runtime_error("dtype needs to be 32-bit integers for indices");
    }
    num_indices = info.indices.num_elements;
    draw_num_indices = info.draw_index_count;
    draw_first_indices = info.draw_first_index % num_indices;
    ibo_size_bytes = num_indices * sizeof(uint32_t);
    if (draw_num_indices + draw_first_indices > num_indices) {
      throw std::runtime_error(
          fmt::format("Requested to draw more indices (#draw_index = {}, "
                      "first_index = {}) than provided ({})",
                      draw_num_indices, draw_first_indices, num_indices));
    }
  }
  // Update render configurations
  config_.vertices_count = int(num_vertices);
  config_.indices_count = num_indices;
  if (info.has_user_customized_draw) {
    config_.draw_vertex_count = draw_num_vertices;
    config_.draw_first_vertex = draw_first_vertices;
    config_.draw_index_count = draw_num_indices;
    config_.draw_first_index = draw_first_indices;
  } else {
    config_.draw_vertex_count = num_vertices;
    config_.draw_first_vertex = 0;
    config_.draw_index_count = num_indices;
    config_.draw_first_index = 0;
  }
  if (num_vertices > max_vertices_count || num_indices > max_indices_count) {
    max_vertices_count = num_vertices;
    max_indices_count = num_indices;
    init_buffers();
  }
  // 3. Copy data
  // If data source is not a host mapped pointer, it is a DeviceAllocation
  // from the same backend as GGUI
  {
    DevicePtr src_ptr = info.vbo.dev_alloc.get_ptr();
    copy_helper(prog, vertex_buffer_->get_ptr(0), src_ptr,
                staging_vertex_buffer_->get_ptr(), vbo_size_bytes);
  }
  if (info.indices.valid) {
    indexed_ = true;
    DevicePtr src_ptr = info.indices.dev_alloc.get_ptr();
    copy_helper(prog, index_buffer_->get_ptr(), src_ptr,
                staging_index_buffer_->get_ptr(), ibo_size_bytes);
  }
}

void Renderable::update_scene_data(DevicePtr ssbo_ptr, DevicePtr ubo_ptr) {
}

Pipeline &Renderable::pipeline() {
  return *pipeline_;
}

const Pipeline &Renderable::pipeline() const {
  return *pipeline_;
}

void Renderable::create_graphics_pipeline() {
  pipeline_ = app_context_->get_raster_pipeline(
      {config_.fragment_shader_path, config_.vertex_shader_path,
       config_.topology_type, config_.depth, config_.polygon_mode,
       config_.blending, config_.vertex_input_rate_instance});
}

void Renderable::record_this_frame_commands(CommandList *command_list) {
  if (uniform_buffer_renderable_) {
    resource_set_->buffer(0, uniform_buffer_renderable_->get_ptr(0));
  }

  auto raster_state = app_context_->device().create_raster_resources_unique();
  if (vertex_buffer_) {
    raster_state->vertex_buffer(vertex_buffer_->get_ptr(0), 0);
  }
  if (index_buffer_) {
    raster_state->index_buffer(index_buffer_->get_ptr(0), 32);
  }

  command_list->bind_pipeline(pipeline_);
  command_list->bind_raster_resources(raster_state.get());
  command_list->bind_shader_resources(resource_set_.get());

  if (indexed_) {
    TI_ASSERT(index_buffer_);
    command_list->draw_indexed(config_.draw_index_count,
                               config_.draw_first_vertex,
                               config_.draw_first_index);
  } else {
    command_list->draw(config_.draw_vertex_count, config_.draw_first_vertex);
  }
}

}  // namespace vulkan

}  // namespace taichi::ui
