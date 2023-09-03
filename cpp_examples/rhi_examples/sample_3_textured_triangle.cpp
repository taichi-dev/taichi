#ifdef RHI_EXAMPLE_BACKEND_VULKAN
#include "common_vulkan.h"
#endif  // RHI_EXAMPLE_BACKEND_VULKAN

#ifdef RHI_EXAMPLE_BACKEND_METAL
#include "common_metal.h"
#endif

std::vector<uint32_t> frag_spv =
#include "shaders/3_triangle.frag.spv.h"
    ;

std::vector<uint32_t> vert_spv =
#include "shaders/3_triangle.vert.spv.h"
    ;

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;
  glm::vec2 texcoord;
};

const std::vector<Vertex> vertices = {
    {{0.0, 0.5}, {1.0, 0.0, 0.0}, {0.5, 0.0}},
    {{0.5, -0.5}, {0.0, 1.0, 0.0}, {1.0, 1.0}},
    {{-0.5, -0.5}, {0.0, 0.0, 1.0}, {0.0, 1.0}},
};

struct UBO {
  float scale;
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
           /*   offset = */ offsetof(Vertex, color)},
          {/* location = */ 2,
           /*  binding = */ 0,
           /*   format = */ BufferFormat::rg32f,
           /*   offset = */ offsetof(Vertex, texcoord)}};

      // Create pipeline
      pipeline = device->create_raster_pipeline(src_desc, raster_params,
                                                vertex_inputs, vertex_attrs);
    }

    // Create a little UBO
    {
      UBO ubo;
      ubo.scale = 1.5;

      // Allocate memory on GPU, and get a DeviceAllocation representing it.
      auto [buf, res] = device->allocate_memory_unique(
          {sizeof(UBO), /*host_write=*/true, /*host_read=*/false,
           /*export_sharing=*/false, AllocUsage::Uniform});
      TI_ASSERT(res == RhiResult::success);

      // Move ownership of DeviceAllocation to our variable.
      uniform_buffer = std::move(buf);

      // Pointer to some place in host memory.
      void *mapped{nullptr};

      // Map the allocation on host to the allocation on GPU.
      // Tells the RHI to get the mapped region in host memory
      // for our allocated buffer on the GPU, and set the
      // mapped pointer to point to this.
      device->map(uniform_buffer->get_ptr(0), &mapped);

      // Copy our data into the mapped place in host memory.
      memcpy(mapped, &ubo, sizeof(ubo));

      // Unmap
      device->unmap(*uniform_buffer);
    }

    // Create our little texture
    {
      constexpr uint32_t tex_size = 256;

      // Just a little 8 bit, 256x256 worley texture
      std::vector<std::pair<double, double>> random_points;
      for (int i = 0; i < 100; i++) {
        double x = tex_size * static_cast<double>(std::rand()) / RAND_MAX;
        double y = tex_size * static_cast<double>(std::rand()) / RAND_MAX;
        random_points.push_back(std::make_pair(x, y));
      }

      size_t tex_data_size = sizeof(unsigned char) * tex_size * tex_size;
      std::vector<unsigned char> vec_data;
      for (int i = 0; i < tex_size * tex_size; i++) {
        double x = (double)(i % 256);
        double y = (double)(i / 256);

        double min_dist = 99999.0;
        for (auto &pair : random_points) {
          double diff_x = x - pair.first;
          double diff_y = y - pair.second;
          min_dist = std::min(min_dist, diff_x * diff_x + diff_y * diff_y);
        }
        min_dist = std::min(std::sqrt(min_dist) / 30.0, 1.0);
        vec_data.push_back((unsigned char)(255 - min_dist * 255));
      }

      // Upload vector image data to buffer
      auto [buf, res] = device->allocate_memory_unique(
          {tex_data_size, /*host_write=*/true,
           /*host_read=*/false,
           /*export_sharing=*/false, AllocUsage::None});
      TI_ASSERT(res == RhiResult::success);
      void *mapped{nullptr};
      device->map(buf->get_ptr(0), &mapped);
      memcpy(mapped, vec_data.data(), tex_data_size);
      device->unmap(*buf);

      // Create the image
      ImageParams params;
      params.dimension = ImageDimension::d2D;
      params.format = BufferFormat::r8;
      params.initial_layout = ImageLayout::undefined;
      params.x = tex_size;
      params.y = tex_size;
      params.z = 1;
      params.export_sharing = false;

      texture = device->create_image_unique(params);

      // Create a command list. Use this cmd list to submit commands to copy the
      // buffer data to the image.
      auto [cmdlist, res2] =
          device->get_graphics_stream()->new_command_list_unique();
      TI_ASSERT(res2 == RhiResult::success);

      // Transition our image to the transfer destination state so we can write
      // to it.
      cmdlist->image_transition(*texture, ImageLayout::undefined,
                                ImageLayout::transfer_dst);
      // Pipeline barrier to define a memory dependency between for the commands
      // accessing the buffer now.
      cmdlist->buffer_barrier(*buf);

      // Copy buffer data to the image.
      BufferImageCopyParams copy_params;
      copy_params.image_extent.x = tex_size;
      copy_params.image_extent.y = tex_size;
      cmdlist->buffer_to_image(*texture, buf->get_ptr(),
                               ImageLayout::transfer_dst, copy_params);

      // Transition the image to the the shader read state.
      cmdlist->image_transition(*texture, ImageLayout::transfer_dst,
                                ImageLayout::shader_read);

      // Submit the cmd list.
      auto transfer_complete_semaphore =
          device->get_graphics_stream()->submit_synced(cmdlist.get(), {});

      // We don't need to keep this semaphore around, since all our work is
      // being done on the graphics stream. So the submit_synced is enough to
      // ensure the GPU work on our image is completed before rendering starts.
    }

    // Define the shader resources
    {
      shader_resources = device->create_resource_set_unique();
      shader_resources->buffer(3, uniform_buffer->get_ptr(0));
      shader_resources->image(5, *texture, {});

      // Using weird binding indices for testing purposes.
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

    // Define the raster resources
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

    res = cmdlist->bind_shader_resources(shader_resources.get());

    TI_ASSERT_INFO(res == RhiResult::success,
                   "Shader res bind fault: RhiResult({})", res);
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
  std::unique_ptr<ShaderResourceSet> shader_resources{nullptr};

  std::unique_ptr<DeviceAllocationGuard> vertex_buffer{nullptr};
  std::unique_ptr<DeviceAllocationGuard> uniform_buffer{nullptr};
  taichi::lang::DeviceImageUnique texture{nullptr};
};

int main() {
  std::unique_ptr<SampleApp> app = std::make_unique<SampleApp>();
  app->run();

  return 0;
}
