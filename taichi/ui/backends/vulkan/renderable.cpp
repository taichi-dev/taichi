#include "taichi/ui/backends/vulkan/renderable.h"
#include "taichi/ui/utils/utils.h"

#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/backends/vulkan/renderer.h"
#include "taichi/ui/backends/vulkan/renderables/kernels.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;


void Renderable::init(const RenderableConfig &config,
                      class Renderer *renderer) {
  config_ = config;
  renderer_ = renderer;
  app_context_ = &renderer->app_context();
}

void Renderable::init_render_resources() {


  create_graphics_pipeline();


  create_vertex_buffer();
  create_index_buffer();
  create_uniform_buffers();
  create_storage_buffers();

  create_bindings();


  if (app_context_->config.ti_arch == Arch::cuda) {
    auto [vb_mem,vb_offset,vb_size] = app_context_->vulkan_device().get_vkmemory_offset_size(vertex_buffer_);

    auto [ib_mem,ib_offset,ib_size] = app_context_->vulkan_device().get_vkmemory_offset_size(index_buffer_);
    
    auto block_size = VulkanDevice::kMemoryBlockSize;

    vertex_buffer_device_ptr_ = (Vertex *)get_memory_pointer(
        vb_mem,block_size,vb_offset,vb_size,
        app_context_->device());
    index_buffer_device_ptr_ = (int *)get_memory_pointer(
        ib_mem,block_size,ib_offset,ib_size,
        app_context_->device());
  }
}

void Renderable::update_data(const RenderableInfo &info) {
  int num_vertices = info.vertices.shape[0];
  int num_indices;
  if (info.indices.valid) {
    num_indices = info.indices.shape[0];
    if (info.indices.dtype != PrimitiveType::i32 &&
        info.indices.dtype != PrimitiveType::u32) {
      throw std::runtime_error("dtype needs to be 32-bit ints for indices");
    }
  } else {
    num_indices = num_vertices;
  }
  if (num_vertices > config_.vertices_count ||
      num_indices > config_.indices_count) {
    cleanup();
    config_.vertices_count = num_vertices;
    config_.indices_count = num_indices;
    init_render_resources();
  }

  if (info.vertices.dtype != PrimitiveType::f32) {
    throw std::runtime_error("dtype needs to be f32 for vertices");
  }

  int num_components = info.vertices.matrix_rows;

  if (info.vertices.field_source == FieldSource::TaichiCuda) {
    update_renderables_vertices_cuda(vertex_buffer_device_ptr_,
                                     (float *)info.vertices.data, num_vertices,
                                     num_components);

    if (info.per_vertex_color.valid) {
      if (info.per_vertex_color.shape[0] != num_vertices) {
        throw std::runtime_error(
            "shape of per_vertex_color should be the same as vertices");
      }
      update_renderables_colors_cuda(vertex_buffer_device_ptr_,
                                     (float *)info.per_vertex_color.data,
                                     num_vertices);
    }

    if (info.normals.valid) {
      if (info.normals.shape[0] != num_vertices) {
        throw std::runtime_error(
            "shape of normals should be the same as vertices");
      }
      update_renderables_normals_cuda(vertex_buffer_device_ptr_,
                                      (float *)info.normals.data, num_vertices);
    }

    if (info.indices.valid) {
      indexed_ = true;
      update_renderables_indices_cuda(index_buffer_device_ptr_,
                                      (int *)info.indices.data, num_indices);
    } else {
      indexed_ = false;
    }

  } else if (info.vertices.field_source == FieldSource::TaichiX64) {
    {
      Vertex* mapped_vbo = (Vertex*)app_context_->vulkan_device().map(staging_vertex_buffer_);
      
      update_renderables_vertices_x64(mapped_vbo,
                                      (float *)info.vertices.data, num_vertices,
                                      num_components);
      if (info.per_vertex_color.valid) {
        if (info.per_vertex_color.shape[0] != num_vertices) {
          throw std::runtime_error(
              "shape of per_vertex_color should be the same as vertices");
        }
        update_renderables_colors_x64(mapped_vbo,
                                      (float *)info.per_vertex_color.data,
                                      num_vertices);
      }
      if (info.normals.valid) {
        if (info.normals.shape[0] != num_vertices) {
          throw std::runtime_error(
              "shape of normals should be the same as vertices");
        }
        update_renderables_normals_x64(mapped_vbo,
                                       (float *)info.normals.data,
                                       num_vertices);
      }
      app_context_->vulkan_device().unmap(staging_vertex_buffer_);

      int* mapped_ibo = (int*)app_context_->vulkan_device().map(staging_index_buffer_);
      if (info.indices.valid) {
        indexed_ = true;
        update_renderables_indices_x64(mapped_ibo,
                                       (int *)info.indices.data, num_indices);
      } else {
        indexed_ = false;
      }
      app_context_->vulkan_device().unmap(staging_index_buffer_);
    }
    app_context_->vulkan_device().memcpy(vertex_buffer_.get_ptr(0),staging_vertex_buffer_.get_ptr(0),config_.vertices_count * sizeof(Vertex));
    app_context_->vulkan_device().memcpy(index_buffer_.get_ptr(0),staging_index_buffer_.get_ptr(0),config_.indices_count * sizeof(int));
  } else {
    throw std::runtime_error("unsupported field source");
  }
}


VulkanPipeline& Renderable::pipeline(){
  return *(pipeline_.get());
}

const VulkanPipeline& Renderable::pipeline() const{
  return *(pipeline_.get());
}

void Renderable::create_bindings(){
  ResourceBinder* binder = pipeline_->resource_binder();
  binder->vertex_buffer(vertex_buffer_.get_ptr(0),0);
  binder->index_buffer(index_buffer_.get_ptr(0),32);
}

void Renderable::create_graphics_pipeline() {
  auto vert_code = read_file(config_.vertex_shader_path);
  auto frag_code = read_file(config_.fragment_shader_path);

  SpirvCodeView vert_view;
  vert_view.data = (uint32_t*)vert_code.data();
  vert_view.size = vert_code.size();
  vert_view.stage = VK_SHADER_STAGE_VERTEX_BIT;

  SpirvCodeView frag_view;
  frag_view.data = (uint32_t*)frag_code.data();
  frag_view.size = frag_code.size();
  frag_view.stage = VK_SHADER_STAGE_FRAGMENT_BIT;

  VulkanPipeline::Params params;
  params.device = &(renderer_->app_context().vulkan_device());
  params.code = {vert_view,frag_view};

  VulkanPipeline::RasterParams raster_params;
  if (config_.topology_type == TopologyType::Triangles) {
    raster_params.prim_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  } else if (config_.topology_type == TopologyType::Lines) {
    raster_params.prim_topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  } else if (config_.topology_type == TopologyType::Points) {
    raster_params.prim_topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  } else {
    throw std::runtime_error("invalid topology");
  }
  raster_params.depth_test = true;
  raster_params.depth_write = true;

  std::vector<VertexInputBinding> vertex_inputs = {{0,sizeof(Vertex),false}};
  std::vector<VertexInputAttribute> vertex_attribs = {
    {0,0,BufferFormat::rgb32f,offsetof(Vertex, pos)},
    {1,0,BufferFormat::rgb32f,offsetof(Vertex, normal)},
    {2,0,BufferFormat::rg32f,offsetof(Vertex, texCoord)},
    {3,0,BufferFormat::rgb32f,offsetof(Vertex, color)}
  };
  
  pipeline_ = std::make_unique<VulkanPipeline>(params,raster_params,vertex_inputs,vertex_attribs);

}

void Renderable::create_vertex_buffer() {
  size_t buffer_size = sizeof(Vertex) * config_.vertices_count;


  Device::AllocParams vb_params {buffer_size,false,false,true,AllocUsage::Vertex};
  vertex_buffer_ = app_context_->vulkan_device().allocate_memory(vb_params);

  Device::AllocParams staging_vb_params {buffer_size,true,false,false,AllocUsage::Vertex};
  staging_vertex_buffer_ = app_context_->vulkan_device().allocate_memory(staging_vb_params);
 
}

void Renderable::create_index_buffer() {
  size_t buffer_size = sizeof(int) * config_.indices_count;

  Device::AllocParams ib_params {buffer_size,false,false,true,AllocUsage::Index};
  index_buffer_ = app_context_->vulkan_device().allocate_memory(ib_params);

  Device::AllocParams staging_ib_params {buffer_size,true,false,false,AllocUsage::Index};
  staging_index_buffer_ = app_context_->vulkan_device().allocate_memory(staging_ib_params);
 
}

void Renderable::create_uniform_buffers() {
  size_t buffer_size = config_.ubo_size;
  if (buffer_size == 0) {
    return;
  }

  Device::AllocParams ub_params {buffer_size,true,false,false,AllocUsage::Uniform};
  uniform_buffer_ = app_context_->vulkan_device().allocate_memory(ub_params);
  
}

void Renderable::create_storage_buffers() {
  size_t buffer_size = config_.ssbo_size;
  if (buffer_size == 0) {
    return;
  } 
  
  Device::AllocParams sb_params {buffer_size,true,false,false,AllocUsage::Storage};
  storage_buffer_ = app_context_->vulkan_device().allocate_memory(sb_params); 
}

void Renderable::destroy_uniform_buffers() {
  if (config_.ubo_size == 0) {
    return;
  }
  app_context_->vulkan_device().dealloc_memory(uniform_buffer_);
}

void Renderable::destroy_storage_buffers() {
  if (config_.ssbo_size == 0) {
    return;
  } 
  app_context_->vulkan_device().dealloc_memory(storage_buffer_);
}


void Renderable::cleanup() {
  
}

void Renderable::record_this_frame_commands(CommandList* command_list) {
  command_list->bind_pipeline(pipeline_.get());
  command_list->bind_resources(pipeline_->resource_binder());

  if (indexed_) {
    command_list->draw_indexed(config_.indices_count,0,0);
  } else {
    command_list->draw(config_.vertices_count,0);
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
