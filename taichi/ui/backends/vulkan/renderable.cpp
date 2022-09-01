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
  TI_ASSERT(info.vbo_attrs == config_.vbo_attrs);
  // We might not have a current program if GGUI is used in external apps to
  // load AOT modules
  Program *prog = app_context_->prog();
  if (prog) {
    prog->flush();
  }

  bool needs_pipeline_reset = false;

  // Check if we need to update Graphics Pipeline
  if (info.display_mode != config_.polygon_mode) {
    needs_pipeline_reset = true;
    config_.polygon_mode = info.display_mode;
    pipeline_.reset();
    create_graphics_pipeline();
  }

  int num_vertices = info.vbo.shape[0];
  int draw_num_vertices = info.draw_vertex_count;
  int draw_first_vertices = info.draw_first_vertex % num_vertices;

  int num_indices;
  int draw_num_indices;
  int draw_first_indices;

  if (info.indices.valid) {
    TI_ERROR_IF(info.indices.matrix_cols != 1,
                "indices must either be a ti.field or a 2D/3D ti.Vector.field");
    num_indices = info.indices.shape[0] * info.indices.matrix_rows;
    draw_num_indices = info.draw_index_count * info.indices.matrix_rows;
    draw_first_indices =
        (info.draw_first_index * info.indices.matrix_rows) % num_indices;
    if (info.indices.dtype != PrimitiveType::i32 &&
        info.indices.dtype != PrimitiveType::u32) {
      throw std::runtime_error("dtype needs to be 32-bit ints for indices");
    }
  } else {
    num_indices = 1;
    draw_num_indices = 1;
    draw_first_indices = 0;
  }

  config_.vertices_count = num_vertices;
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

  if (needs_pipeline_reset || num_vertices > config_.max_vertices_count ||
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

  const uint64_t vbo_size = config_.vbo_size() * num_vertices;

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
    DevicePtr ibo_dev_ptr = info.indices.dev_alloc.get_ptr();
    if (prog) {
      ibo_dev_ptr = get_device_ptr(prog, info.indices.snode);
    }
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
  raster_params.polygon_mode = config_.polygon_mode;
  raster_params.depth_test = true;
  raster_params.depth_write = true;

  if (config_.blending) {
    raster_params.blending.push_back(BlendingParams());
  }

  std::vector<VertexInputBinding> vertex_inputs = {
      {/*binding=*/0, config_.vbo_size(), /*instance=*/false}};
  // TODO: consider using uint8 for colors and normals
  std::vector<VertexInputAttribute> vertex_attribs;
  if (VboHelpers::has_attr(config_.vbo_attrs, VertexAttributes::kPos)) {
    vertex_attribs.push_back({/*location=*/0, /*binding=*/0,
                              /*format=*/BufferFormat::rgb32f,
                              /*offset=*/offsetof(Vertex, pos)});
  }
  if (VboHelpers::has_attr(config_.vbo_attrs, VertexAttributes::kNormal)) {
    vertex_attribs.push_back({/*location=*/1, /*binding=*/0,
                              /*format=*/BufferFormat::rgb32f,
                              /*offset=*/offsetof(Vertex, normal)});
  }
  if (VboHelpers::has_attr(config_.vbo_attrs, VertexAttributes::kUv)) {
    vertex_attribs.push_back({/*location=*/2, /*binding=*/0,
                              /*format=*/BufferFormat::rg32f,
                              /*offset=*/offsetof(Vertex, tex_coord)});
  }
  if (VboHelpers::has_attr(config_.vbo_attrs, VertexAttributes::kColor)) {
    vertex_attribs.push_back({/*location=*/3, /*binding=*/0,
                              /*format=*/BufferFormat::rgba32f,
                              /*offset=*/offsetof(Vertex, color)});
  }

  pipeline_ = app_context_->device().create_raster_pipeline(
      source, raster_params, vertex_inputs, vertex_attribs);
}

void Renderable::create_vertex_buffer() {
  const size_t buffer_size = config_.vbo_size() * config_.max_vertices_count;

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
  const size_t buffer_size = config_.ubo_size;
  if (buffer_size == 0) {
    return;
  }

  Device::AllocParams ub_params{buffer_size, true, false, false,
                                AllocUsage::Uniform};
  uniform_buffer_ = app_context_->device().allocate_memory(ub_params);
}

void Renderable::create_storage_buffers() {
  const size_t buffer_size = config_.ssbo_size;
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
    command_list->draw_indexed(config_.draw_index_count,
                               config_.draw_first_vertex,
                               config_.draw_first_index);
  } else {
    command_list->draw(config_.draw_vertex_count, config_.draw_first_vertex);
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
