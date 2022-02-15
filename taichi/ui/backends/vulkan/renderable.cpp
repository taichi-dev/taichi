#include "taichi/ui/backends/vulkan/renderable.h"
#include "taichi/program/program.h"
#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void Renderable::init(const RenderableConfig &config, AppContext *app_context) {
  config_ = config;
  app_context_ = app_context;
}

void Renderable::init_render_resources() {
  create_graphics_pipeline();
  init_buffers();
}

void Renderable::free_buffers() {
  app_context_->device().dealloc_memory(vertex_buffer_);
  app_context_->device().dealloc_memory(staging_vertex_buffer_);
  app_context_->device().dealloc_memory(index_buffer_);
  app_context_->device().dealloc_memory(staging_index_buffer_);

  destroy_uniform_buffers();
  destroy_storage_buffers();
}

void Renderable::init_buffers() {
  create_vertex_buffer();
  create_index_buffer();
  create_uniform_buffers();
  create_storage_buffers();

  create_bindings();
}

void Renderable::update_data(const RenderableInfo &info) {
  // We might not have a current program if GGUI is used in external apps to
  // load AOT modules
  Program *prog = app_context_->prog();
  if (prog) {
    prog->synchronize();
  }

  int num_vertices = info.vbo.shape[0];
  int num_indices;
  if (info.indices.valid) {
    TI_ERROR_IF(info.indices.matrix_cols != 1,
                "indices must either be a ti.field or a 2D/3D ti.Vector.field");
    num_indices = info.indices.shape[0] * info.indices.matrix_rows;
    if (info.indices.dtype != PrimitiveType::i32 &&
        info.indices.dtype != PrimitiveType::u32) {
      throw std::runtime_error("dtype needs to be 32-bit ints for indices");
    }
  } else {
    num_indices = 1;
  }

  config_.vertices_count = num_vertices;
  config_.indices_count = num_indices;

  if (num_vertices > config_.max_vertices_count ||
      num_indices > config_.max_indices_count) {
    free_buffers();
    config_.max_vertices_count = num_vertices;
    config_.max_indices_count = num_indices;
    init_buffers();
  }

  // If there is no current program, VBO information should be provided directly
  // instead of accessing through the current SNode
  DevicePtr vbo_dev_ptr = info.vbo.dev_alloc.get_ptr();
  if (prog) {
    vbo_dev_ptr = get_device_ptr(prog, info.vbo.snode);
  }

  uint64_t vbo_size = sizeof(Vertex) * num_vertices;

  Device::MemcpyCapability memcpy_cap = Device::check_memcpy_capability(
      vertex_buffer_.get_ptr(), vbo_dev_ptr, vbo_size);
  if (memcpy_cap == Device::MemcpyCapability::Direct) {
    Device::memcpy_direct(vertex_buffer_.get_ptr(), vbo_dev_ptr, vbo_size);
  } else if (memcpy_cap == Device::MemcpyCapability::RequiresStagingBuffer) {
    Device::memcpy_via_staging(vertex_buffer_.get_ptr(),
                               staging_vertex_buffer_.get_ptr(), vbo_dev_ptr,
                               vbo_size);
  } else {
    TI_NOT_IMPLEMENTED;
  }

  if (info.indices.valid) {
    indexed_ = true;
    DevicePtr ibo_dev_ptr = get_device_ptr(prog, info.indices.snode);
    uint64_t ibo_size = num_indices * sizeof(int);
    if (memcpy_cap == Device::MemcpyCapability::Direct) {
      Device::memcpy_direct(index_buffer_.get_ptr(), ibo_dev_ptr, ibo_size);
    } else if (memcpy_cap == Device::MemcpyCapability::RequiresStagingBuffer) {
      Device::memcpy_via_staging(index_buffer_.get_ptr(),
                                 staging_index_buffer_.get_ptr(), ibo_dev_ptr,
                                 ibo_size);
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }
}

Pipeline &Renderable::pipeline() {
  return *(pipeline_.get());
}

const Pipeline &Renderable::pipeline() const {
  return *(pipeline_.get());
}

void Renderable::create_bindings() {
  ResourceBinder *binder = pipeline_->resource_binder();
  binder->vertex_buffer(vertex_buffer_.get_ptr(0), 0);
  binder->index_buffer(index_buffer_.get_ptr(0), 32);
}

void Renderable::create_graphics_pipeline() {
  if (pipeline_.get()) {
    return;
  }
  auto vert_code = read_file(config_.vertex_shader_path);
  auto frag_code = read_file(config_.fragment_shader_path);

  std::vector<PipelineSourceDesc> source(2);
  source[0] = {PipelineSourceType::spirv_binary, frag_code.data(),
               frag_code.size(), PipelineStageType::fragment};
  source[1] = {PipelineSourceType::spirv_binary, vert_code.data(),
               vert_code.size(), PipelineStageType::vertex};

  RasterParams raster_params;
  raster_params.prim_topology = config_.topology_type;
  raster_params.depth_test = true;
  raster_params.depth_write = true;

  std::vector<VertexInputBinding> vertex_inputs = {{0, sizeof(Vertex), false}};
  // TODO: consider using uint8 for colors and normals
  std::vector<VertexInputAttribute> vertex_attribs = {
      {0, 0, BufferFormat::rgb32f, offsetof(Vertex, pos)},
      {1, 0, BufferFormat::rgb32f, offsetof(Vertex, normal)},
      {2, 0, BufferFormat::rg32f, offsetof(Vertex, texCoord)},
      {3, 0, BufferFormat::rgba32f, offsetof(Vertex, color)}};

  pipeline_ = app_context_->device().create_raster_pipeline(
      source, raster_params, vertex_inputs, vertex_attribs);
}

void Renderable::create_vertex_buffer() {
  size_t buffer_size = sizeof(Vertex) * config_.max_vertices_count;

  Device::AllocParams vb_params{buffer_size, false, false,
                                app_context_->requires_export_sharing(),
                                AllocUsage::Vertex};
  vertex_buffer_ = app_context_->device().allocate_memory(vb_params);

  Device::AllocParams staging_vb_params{buffer_size, true, false, false,
                                        AllocUsage::Vertex};
  staging_vertex_buffer_ =
      app_context_->device().allocate_memory(staging_vb_params);
}

void Renderable::create_index_buffer() {
  size_t buffer_size = sizeof(int) * config_.max_indices_count;

  Device::AllocParams ib_params{buffer_size, false, false,
                                app_context_->requires_export_sharing(),
                                AllocUsage::Index};
  index_buffer_ = app_context_->device().allocate_memory(ib_params);

  Device::AllocParams staging_ib_params{buffer_size, true, false, false,
                                        AllocUsage::Index};
  staging_index_buffer_ =
      app_context_->device().allocate_memory(staging_ib_params);
}

void Renderable::create_uniform_buffers() {
  size_t buffer_size = config_.ubo_size;
  if (buffer_size == 0) {
    return;
  }

  Device::AllocParams ub_params{buffer_size, true, false, false,
                                AllocUsage::Uniform};
  uniform_buffer_ = app_context_->device().allocate_memory(ub_params);
}

void Renderable::create_storage_buffers() {
  size_t buffer_size = config_.ssbo_size;
  if (buffer_size == 0) {
    return;
  }

  Device::AllocParams sb_params{buffer_size, true, false, false,
                                AllocUsage::Storage};
  storage_buffer_ = app_context_->device().allocate_memory(sb_params);
}

void Renderable::destroy_uniform_buffers() {
  if (config_.ubo_size == 0) {
    return;
  }
  app_context_->device().dealloc_memory(uniform_buffer_);
}

void Renderable::destroy_storage_buffers() {
  if (config_.ssbo_size == 0) {
    return;
  }
  app_context_->device().dealloc_memory(storage_buffer_);
}

void Renderable::cleanup() {
  free_buffers();
  pipeline_.reset();
}

void Renderable::record_this_frame_commands(CommandList *command_list) {
  command_list->bind_pipeline(pipeline_.get());
  command_list->bind_resources(pipeline_->resource_binder());

  if (indexed_) {
    command_list->draw_indexed(config_.indices_count, 0, 0);
  } else {
    command_list->draw(config_.vertices_count, 0);
  }
}

void Renderable::resize_storage_buffers(int new_ssbo_size) {
  if (new_ssbo_size == config_.ssbo_size) {
    return;
  }
  destroy_storage_buffers();
  config_.ssbo_size = new_ssbo_size;
  create_storage_buffers();
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
