#include "common.h"

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
      std::vector<VertexInputBinding> vertex_inputs = {
          {.binding = 0, .stride = sizeof(Vertex), .instance = false}};
      std::vector<VertexInputAttribute> vertex_attrs = {
          {.location = 0,
           .binding = 0,
           .format = BufferFormat::rg32f,
           .offset = offsetof(Vertex, pos)},
          {.location = 1,
           .binding = 0,
           .format = BufferFormat::rgb32f,
           .offset = offsetof(Vertex, color)}};

      // Create pipeline
      pipeline = device->create_raster_pipeline(src_desc, raster_params,
                                                vertex_inputs, vertex_attrs);
    }

    // Create the vertex buffer
    {
      vertex_buffer = device->allocate_memory_unique(
          Device::AllocParams{.size = 3 * sizeof(Vertex),
                              .host_write = true,
                              .usage = AllocUsage::Vertex});
      Vertex *mapped = (Vertex *)device->map(*vertex_buffer);
      mapped[0] = {{0.0, 0.5}, {1.0, 0.0, 0.0}};
      mapped[1] = {{0.5, -0.5}, {0.0, 1.0, 0.0}};
      mapped[2] = {{-0.5, -0.5}, {0.0, 0.0, 1.0}};
      device->unmap(*vertex_buffer);
    }

    TI_INFO("App Init Done");
  }

  std::vector<StreamSemaphore> render_loop(
      StreamSemaphore image_available_semaphore) override {
    auto cmdlist = device->get_graphics_stream()->new_command_list();

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
    // Get the binder and bind our vertex buffer
    auto resource_binder = pipeline->resource_binder();
    resource_binder->vertex_buffer(vertex_buffer->get_ptr(0), 0);
    cmdlist->bind_resources(resource_binder);
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
  std::unique_ptr<Pipeline> pipeline;

  std::unique_ptr<DeviceAllocationGuard> vertex_buffer;
};

int main() {
  std::unique_ptr<SampleApp> app = std::make_unique<SampleApp>();
  app->run();

  return 0;
}
