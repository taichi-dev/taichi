#ifdef RHI_EXAMPLE_BACKEND_VULKAN
#include "common_vulkan.h"
#endif  // RHI_EXAMPLE_BACKEND_VULKAN

#ifdef RHI_EXAMPLE_BACKEND_METAL
#include "common_metal.h"
#endif

std::vector<uint32_t> frag_spv =
#include "shaders/2_triangle.frag.spv.h"
    ;

std::vector<uint32_t> vert_spv =
#include "shaders/2_triangle.vert.spv.h"
    ;

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;
};

const std::vector<Vertex> vertices = {
    {{0.0, 0.5}, {1.0, 0.0, 0.0}},
    {{0.5, -0.5}, {0.0, 1.0, 0.0}},
    {{-0.5, -0.5}, {0.0, 0.0, 1.0}},
};

class SampleApp : public App {
 public:
  SampleApp() : App(1920, 1080, "Sample 2: Triangle") {
    // Create the triangle raster pipeline
    {
      // Load the SPIRV source
      std::vector<PipelineSourceDesc> src_desc(2);

      src_desc[0].data = (void *)frag_spv.data();
      src_desc[0].size = frag_spv.size() * sizeof(uint32_t);
      src_desc[0].type = PipelineSourceType::spirv_binary;
      src_desc[0].stage = PipelineStageType::fragment;

      src_desc[1].data = (void *)vert_spv.data();
      src_desc[1].size = vert_spv.size() * sizeof(uint32_t);
      src_desc[1].type = PipelineSourceType::spirv_binary;
      src_desc[1].stage = PipelineStageType::vertex;

      // Setup rasterizer parameters
      RasterParams raster_params;  // use default

      // Setup vertex input parameters
      // FIXME: Switch to designated initializers when we enable C++20
      std::vector<VertexInputBinding> vertex_inputs = {
          {/* binding = */ 0, /* stride = */ sizeof(Vertex),
           /* instance = */ false}};
      std::vector<VertexInputAttribute> vertex_attrs = {
          {/* location = */ 0,
           /*  binding = */ 0,
           /*   format = */ BufferFormat::rg32f,
           /*   offset = */ offsetof(Vertex, pos)},
          {/* location = */ 1,
           /*  binding = */ 0,
           /*   format = */ BufferFormat::rgb32f,
           /*   offset = */ offsetof(Vertex, color)}};

      // Create pipeline
      pipeline = device->create_raster_pipeline(src_desc, raster_params,
                                                vertex_inputs, vertex_attrs);
    }

    // Create the vertex buffer
    {
      auto [buf, res] = device->allocate_memory_unique(Device::AllocParams{
          /* size = */ 3 * sizeof(Vertex),
          /* host_write = */ true,
          /* host_read = */ false, /* export_sharing = */ false,
          /* usage = */ AllocUsage::Vertex});
      TI_ASSERT(res == RhiResult::success);
      vertex_buffer = std::move(buf);
      void *mapped{nullptr};
      TI_ASSERT(device->map(*vertex_buffer, &mapped) == RhiResult::success);
      memcpy(mapped, vertices.data(), sizeof(Vertex) * vertices.size());
      device->unmap(*vertex_buffer);
    }

    // Define the raster state
    {
      raster_resources = device->create_raster_resources_unique();
      raster_resources->vertex_buffer(vertex_buffer->get_ptr(0), 0);
    }

    TI_INFO("App Init Done");
  }

  std::vector<StreamSemaphore> render_loop(
      StreamSemaphore image_available_semaphore) override {
    auto [cmdlist, res] =
        device->get_graphics_stream()->new_command_list_unique();
    TI_ASSERT(res == RhiResult::success);

    // Set-up our frame buffer attachment
    DeviceAllocation surface_image = surface->get_target_image();
    cmdlist->image_transition(surface_image, ImageLayout::undefined,
                              ImageLayout::color_attachment);

    // Renderpass: render to surface image, clear color values
    bool clear = true;
    std::vector<float> clear_color = {0.1, 0.2, 0.3, 1.0};
    const auto &[width, height] = surface->get_size();
    cmdlist->begin_renderpass(0, 0, width, height, 1, &surface_image, &clear,
                              &clear_color, nullptr, false);

    // Bind our triangle pipeline
    cmdlist->bind_pipeline(pipeline.get());
    res = cmdlist->bind_raster_resources(raster_resources.get());
    TI_ASSERT_INFO(res == RhiResult::success,
                   "Raster res bind fault: RhiResult({})", res);
    // Render the triangle
    cmdlist->draw(3, 0);
    // End rendering
    cmdlist->end_renderpass();

    // Submit command list, returns render complete semaphore
    auto render_complete_semaphore = device->get_graphics_stream()->submit(
        cmdlist.get(), {image_available_semaphore});
    return {render_complete_semaphore};
  }

 public:
  std::unique_ptr<Pipeline> pipeline{nullptr};
  std::unique_ptr<RasterResources> raster_resources{nullptr};

  std::unique_ptr<DeviceAllocationGuard> vertex_buffer{nullptr};
};

int main() {
  std::unique_ptr<SampleApp> app = std::make_unique<SampleApp>();
  app->run();

  return 0;
}
