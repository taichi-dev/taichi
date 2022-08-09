#include "c_api/src/gui_utils/gui_helper.h"
#include "taichi/gui/gui.h"
#include "taichi/ui/backends/vulkan/renderer.h"

namespace gui_api {

static taichi::Arch get_taichi_arch(TiArch arch) {
  switch (arch) {
    case TiArch::TI_ARCH_VULKAN: {
      return taichi::Arch::vulkan;
    }
    case TiArch::TI_ARCH_X64: {
      return taichi::Arch::x64;
    }
    case TiArch::TI_ARCH_CUDA: {
      return taichi::Arch::cuda;
    }
    default: {
      throw std::invalid_argument("Unsupported architecture");
    }
  }
}

static taichi::ui::FieldSource get_field_source(TiArch arch) {
  switch (arch) {
    case TiArch::TI_ARCH_VULKAN: {
      return taichi::ui::FieldSource::TaichiVulkan;
    }
    case TiArch::TI_ARCH_X64: {
      return taichi::ui::FieldSource::TaichiX64;
    }
    case TiArch::TI_ARCH_CUDA: {
      return taichi::ui::FieldSource::TaichiCuda;
    }
    default: {
      throw std::invalid_argument("Unsupported architecture");
    }
  }
}

static taichi::lang::DataType get_taichi_lang_dtype(TiDataType dtype) {
  switch (dtype) {
    case TiDataType::TI_DATA_TYPE_F32: {
      return taichi::lang::PrimitiveType::f32;
    }
    case TiDataType::TI_DATA_TYPE_F64: {
      return taichi::lang::PrimitiveType::f64;
    }
    case TiDataType::TI_DATA_TYPE_I32: {
      return taichi::lang::PrimitiveType::i32;
    }
    case TiDataType::TI_DATA_TYPE_I64: {
      return taichi::lang::PrimitiveType::i64;
    }
    case TiDataType::TI_DATA_TYPE_U32: {
      return taichi::lang::PrimitiveType::u32;
    }
    case TiDataType::TI_DATA_TYPE_U16: {
      return taichi::lang::PrimitiveType::u16;
    }
    case TiDataType::TI_DATA_TYPE_I16: {
      return taichi::lang::PrimitiveType::i16;
    }
    case TiDataType::TI_DATA_TYPE_F16: {
      return taichi::lang::PrimitiveType::f16;
    }
    case TiDataType::TI_DATA_TYPE_U8: {
      return taichi::lang::PrimitiveType::u8;
    }
    case TiDataType::TI_DATA_TYPE_I8: {
      return taichi::lang::PrimitiveType::i8;
    }
    default: {
      throw std::invalid_argument("Unsupported TiDataType");
    }
  }
}

static taichi::ui::FieldInfo create_field_info(
    TiArch arch,
    TiDataType dtype,
    const std::vector<int> &shape,
    const taichi::lang::DeviceAllocation &devalloc) {
  taichi::ui::FieldInfo f_info;
  f_info.valid = true;
  f_info.field_type = taichi::ui::FieldType::Scalar;
  f_info.matrix_rows = 1;
  f_info.matrix_cols = 1;
  f_info.shape = shape;

  f_info.field_source = get_field_source(arch);
  f_info.dtype = get_taichi_lang_dtype(dtype);
  f_info.snode = nullptr;
  f_info.dev_alloc = devalloc;

  return f_info;
}

int GuiHelper::set_circle_info(TiArch arch,
                               TiDataType dtype,
                               const std::vector<int> &shape,
                               const taichi::lang::DeviceAllocation &devalloc) {
  auto f_info = create_field_info(arch, dtype, shape, devalloc);

  int handle = 0;
  if (!circle_info_.empty()) {
    int last_handle = circle_info_.rbegin()->first;
    handle = last_handle + 1;
  }

  auto &circles = circle_info_[handle];

  circles.renderable_info.has_per_vertex_color = false;
  circles.renderable_info.vbo_attrs = taichi::ui::VertexAttributes::kPos;
  circles.renderable_info.vbo = std::move(f_info);
  circles.color = {0.8, 0.4, 0.1};
  circles.radius = 0.005f;  // 0.0015f looks unclear on desktop

  return handle;
}

int GuiHelper::set_image_info(TiArch arch,
                              TiDataType dtype,
                              const std::vector<int> &shape,
                              const taichi::lang::DeviceAllocation &devalloc) {
  auto f_info = create_field_info(arch, dtype, shape, devalloc);

  int handle = 0;
  if (!circle_info_.empty()) {
    int last_handle = circle_info_.rbegin()->first;
    handle = last_handle + 1;
  }

  auto &img_info = img_info_[handle];
  img_info.img = std::move(f_info);

  return handle;
}

void GuiHelper::render_circle(int handle) {
  if (circle_info_.count(handle) == 0) {
    throw std::invalid_argument("Unable to find img_info handle");
  }

  renderer_->circles(circle_info_[handle]);
  renderer_->draw_frame(gui_.get());
  renderer_->swap_chain().surface().present_image();
  renderer_->prepare_for_next_frame();

  glfwSwapBuffers(window_);
  glfwPollEvents();
}

void GuiHelper::render_image(int handle) {
  if (img_info_.count(handle) == 0) {
    throw std::invalid_argument("Unable to find img_info handle");
  }

  renderer_->set_image(img_info_[handle]);
  renderer_->draw_frame(gui_.get());
  renderer_->swap_chain().surface().present_image();
  renderer_->prepare_for_next_frame();

  glfwSwapBuffers(window_);
  glfwPollEvents();
}

GuiHelper::GuiHelper(TiArch arch,
                     const char *shader_path,
                     int window_h,
                     int window_w,
                     bool is_packed_mode) {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ = glfwCreateWindow(window_h, window_w, "Taichi show", NULL, NULL);
  if (window_ == NULL) {
    TI_ERROR("Failed to create GLFW window");
    glfwTerminate();
  }

  // Create a GGUI configuration
  taichi::ui::AppConfig app_config;
  app_config.name = "TaichiSparse";
  app_config.width = window_w;
  app_config.height = window_h;
  app_config.vsync = true;
  app_config.show_window = false;
  app_config.package_path = shader_path;  // make it flexible later
  app_config.ti_arch = get_taichi_arch(arch);
  app_config.is_packed_mode = is_packed_mode;

  // Create GUI & renderer
  renderer_ = std::make_unique<taichi::ui::vulkan::Renderer>();
  renderer_->init(nullptr, window_, app_config);

  renderer_->set_background_color({0.6, 0.6, 0.6});

  gui_ = std::make_shared<taichi::ui::vulkan::Gui>(
      &renderer_->app_context(), &renderer_->swap_chain(), window_);
}

GuiHelper::~GuiHelper() {
  img_info_.clear();
  circle_info_.clear();
  gui_.reset();
  renderer_.reset();
}

}  // namespace gui_api
