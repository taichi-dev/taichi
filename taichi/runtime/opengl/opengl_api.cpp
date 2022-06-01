#include "opengl_api.h"

#include <list>

#include "taichi/backends/opengl/opengl_kernel_util.h"
#include "taichi/backends/opengl/opengl_utils.h"
#include "taichi/runtime/opengl/shaders/runtime.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/program/py_print_buffer.h"
#include "taichi/util/environ_config.h"

#ifdef TI_WITH_OPENGL
#include "glad/gl.h"
#include "glad/egl.h"
#include "GLFW/glfw3.h"
#include "taichi/backends/opengl/opengl_device.h"
#endif  // TI_WITH_OPENGL

namespace taichi {
namespace lang {
namespace opengl {

#define PER_OPENGL_EXTENSION(x) bool opengl_extension_##x;
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION

// will later be initialized in initialize_opengl, here we use the minimum
// value according to OpenGL spec in case glGetIntegerv didn't work properly
int opengl_max_block_dim = 1024;
int opengl_max_grid_dim = 1024;

// kUseGles is set at most once in initialize_opengl below.
// TODO: Properly support setting GLES/GLSL in opengl backend
// without this global static boolean.
static bool kUseGles = false;

#ifdef TI_WITH_OPENGL

struct OpenGlRuntimeImpl {
  struct {
    DeviceAllocation runtime = kDeviceNullAllocation;
    DeviceAllocation root = kDeviceNullAllocation;
    DeviceAllocation gtmp = kDeviceNullAllocation;
  } core_bufs;

  OpenGlRuntimeImpl() {
  }

  std::unique_ptr<GLSLRuntime> runtime{nullptr};
  std::vector<std::unique_ptr<DeviceCompiledTaichiKernel>> programs;
};

// TODO: Move this into ProgramImpl class so that it naturally
// gets access to config->use_gles.
bool initialize_opengl(bool use_gles, bool error_tolerance) {
  static std::optional<bool> supported;  // std::nullopt

  TI_TRACE("initialize_opengl({}, {}) called", use_gles, error_tolerance);

  if (supported.has_value()) {  // this function has been called before
    if (supported.value()) {    // detected to be true in last call
      return true;
    } else {
      if (!error_tolerance)  // not called from with_opengl
        TI_ERROR("OpenGL not supported");
      return false;
    }
  }

  // Code below is guaranteed to be called at most once.
  int opengl_version = 0;

  if (glfwInit()) {
    // Compute Shader requires OpenGL 4.3+ (or OpenGL ES 3.1+)
    if (use_gles) {
      glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    } else {
      glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);
    // GL context needs a window (when using GLFW)
    GLFWwindow *window =
        glfwCreateWindow(1, 1, "Make OpenGL Context", nullptr, nullptr);
    if (!window) {
      const char *desc = nullptr;
      int status = glfwGetError(&desc);
      if (!desc)
        desc = "Unknown Error";
      TI_DEBUG("[glsl] cannot create GLFW window: error {}: {}", status, desc);
    } else {
      glfwMakeContextCurrent(window);
      if (use_gles) {
        opengl_version = gladLoadGLES2(glfwGetProcAddress);
      } else {
        opengl_version = gladLoadGL(glfwGetProcAddress);
      }
      TI_DEBUG("OpenGL context loaded through GLFW");
    }
  }

  if (!opengl_version) {
    TI_TRACE("Attempting to load with EGL");

    // Try EGL instead
    int egl_version = gladLoaderLoadEGL(nullptr);

    if (!egl_version) {
      TI_DEBUG("Failed to load EGL");
    } else {
      static const EGLint configAttribs[] = {EGL_SURFACE_TYPE,
                                             EGL_PBUFFER_BIT,
                                             EGL_BLUE_SIZE,
                                             8,
                                             EGL_GREEN_SIZE,
                                             8,
                                             EGL_RED_SIZE,
                                             8,
                                             EGL_DEPTH_SIZE,
                                             8,
                                             EGL_RENDERABLE_TYPE,
                                             EGL_OPENGL_BIT,
                                             EGL_NONE};

      // Initialize EGL
      EGLDisplay egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

      EGLint major, minor;
      eglInitialize(egl_display, &major, &minor);

      egl_version = gladLoaderLoadEGL(egl_display);

      TI_DEBUG("Loaded EGL {}.{} on display {}",
               GLAD_VERSION_MAJOR(egl_version), GLAD_VERSION_MINOR(egl_version),
               egl_display);

      // Select an appropriate configuration
      EGLint num_configs;
      EGLConfig egl_config;

      eglChooseConfig(egl_display, configAttribs, &egl_config, 1, &num_configs);

      // Bind the API (EGL >= 1.2)
      if (egl_version >= GLAD_MAKE_VERSION(1, 2)) {
        eglBindAPI(use_gles ? EGL_OPENGL_ES_API : EGL_OPENGL_API);
      }

      // Create a context and make it current
      EGLContext egl_context = EGL_NO_CONTEXT;
      if (use_gles) {
        static const EGLint gl_attribs[] = {
            EGL_CONTEXT_MAJOR_VERSION,
            3,
            EGL_CONTEXT_MINOR_VERSION,
            1,
            EGL_NONE,
        };

        egl_context = eglCreateContext(egl_display, egl_config, EGL_NO_CONTEXT,
                                       gl_attribs);
      } else {
        egl_context =
            eglCreateContext(egl_display, egl_config, EGL_NO_CONTEXT, nullptr);
      }

      eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, egl_context);

      if (use_gles) {
        opengl_version = gladLoadGLES2(glad_eglGetProcAddress);
      } else {
        opengl_version = gladLoadGL(glad_eglGetProcAddress);
      }
    }
  }

  // Load OpenGL API
  if (!opengl_version) {
    if (error_tolerance) {
      TI_WARN("Can not create OpenGL context");
      supported = std::make_optional<bool>(false);
      return false;
    }
    TI_ERROR("Can not create OpenGL context");
  }

  TI_DEBUG("{} version {}.{}", use_gles ? "GLES" : "OpenGL",
           GLAD_VERSION_MAJOR(opengl_version),
           GLAD_VERSION_MINOR(opengl_version));

#define PER_OPENGL_EXTENSION(x)          \
  if ((opengl_extension_##x = GLAD_##x)) \
    TI_TRACE("[glsl] Found " #x);
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION

  if (!use_gles && !opengl_extension_GL_ARB_compute_shader) {
    if (error_tolerance) {
      TI_INFO("Your OpenGL does not support GL_ARB_compute_shader extension");
      supported = std::make_optional<bool>(false);
      return false;
    }
    TI_ERROR("Your OpenGL does not support GL_ARB_compute_shader extension");
  }

  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &opengl_max_block_dim);
  check_opengl_error("glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT)");
  TI_TRACE("GL_MAX_COMPUTE_WORK_GROUP_COUNT: {}", opengl_max_block_dim);
  glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &opengl_max_grid_dim);
  check_opengl_error("glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE)");
  TI_TRACE("GL_MAX_COMPUTE_WORK_GROUP_SIZE: {}", opengl_max_grid_dim);

  supported = std::make_optional<bool>(true);
  kUseGles = use_gles;
  return true;
}

void CompiledTaichiKernel::init_args(Kernel *kernel) {
  arg_count = kernel->args.size();
  ret_count = 0;
  for (auto &ret : kernel->rets) {
    if (auto tensor_type = ret.dt->cast<TensorType>())
      ret_count += tensor_type->get_num_elements();
    else
      ret_count += 1;
  }
  for (int i = 0; i < arg_count; i++) {
    const auto dtype_name = kernel->args[i].dt.to_string();
    if (kernel->args[i].is_array) {
      constexpr uint64 kUnkownRuntimeSize = 0;
      arr_args[i] = CompiledArrayArg(
          {/*dtype_enum=*/to_gl_dtype_enum(kernel->args[i].dt), dtype_name,
           /*field_dim=*/kernel->args[i].total_dim -
               kernel->args[i].element_shape.size(),
           /*is_scalar=*/kernel->args[i].element_shape.size() == 0,
           /*element_shape=*/kernel->args[i].element_shape,
           /*shape_offset_in_bytes_in_args_buf=*/taichi_opengl_extra_args_base +
               i * taichi_max_num_indices * sizeof(int),
           kUnkownRuntimeSize});
    } else {
      scalar_args[i] = ScalarArg(
          {dtype_name, /*offset_in_bytes_in_args_buf=*/i * sizeof(uint64_t)});
    }
  }

  args_buf_size = arg_count * sizeof(uint64_t);
  if (arr_args.size()) {
    args_buf_size = taichi_opengl_extra_args_base +
                    arg_count * taichi_max_num_indices * sizeof(int);
  }

  ret_buf_size = ret_count * sizeof(uint64_t);
}

void CompiledTaichiKernel::add(
    const std::string &name,
    const std::string &source_code,
    OffloadedTaskType type,
    const std::string &range_hint,
    int num_workgroups,
    int workgroup_size,
    std::unordered_map<int, irpass::ExternalPtrAccess> *ext_ptr_access) {
  num_workgroups = std::min(num_workgroups, opengl_max_grid_dim);
  workgroup_size = std::min(workgroup_size, opengl_max_block_dim);

  size_t layout_pos = source_code.find("precision highp float;\n");
  TI_ASSERT(layout_pos != std::string::npos);
  std::string source =
      source_code.substr(0, layout_pos) +
      fmt::format(
          "layout(local_size_x = {}, local_size_y = 1, local_size_z = "
          "1) in;\n",
          workgroup_size) +
      source_code.substr(layout_pos);

  TI_DEBUG("[glsl]\ncompiling kernel {}<<<{}, {}>>>:\n{}", name, num_workgroups,
           workgroup_size, source);
  tasks.push_back(
      {name, source, type, range_hint, workgroup_size, num_workgroups});

  if (ext_ptr_access) {
    for (auto pair : *ext_ptr_access) {
      if (ext_arr_access.find(pair.first) != ext_arr_access.end()) {
        ext_arr_access[pair.first] = ext_arr_access[pair.first] | pair.second;
      } else {
        ext_arr_access[pair.first] = pair.second;
      }
    }
  }
}

int CompiledTaichiKernel::lookup_or_add_string(const std::string &str) {
  int i;
  for (i = 0; i < str_table.size(); i++) {
    if (str_table[i] == str) {
      return i;
    }
  }
  str_table.push_back(str);
  return i;
}

void dump_message_buffer(Device *device,
                         DeviceAllocation runtime_buf,
                         const std::vector<std::string> &str_table) {
  auto rt_buf = (GLSLRuntime *)device->map(runtime_buf);

  auto msg_count = rt_buf->msg_count;
  if (msg_count > MAX_MESSAGES) {
    TI_WARN("[glsl] Too much print within one kernel: {} > {}, clipping",
            msg_count, MAX_MESSAGES);
    msg_count = MAX_MESSAGES;
  }

  for (int i = 0; i < msg_count; i++) {
    auto const &msg = rt_buf->msg_buf[i];
    for (int j = 0; j < msg.num_contents; j++) {
      int type = msg.get_type_of(j);
      auto value = msg.contents[j];

      std::string str;
      switch (type) {
        case 1:
          str = fmt::format("{}", value.val_i32);
          break;
        case 2:
          str = fmt::format("{}", value.val_f32);
          break;
        case 3:
          str = str_table.at(value.val_i32);
          break;
        default:
          TI_WARN("[glsl] Unexpected serialization type: {}, ignoring", type);
          break;
      };
      py_cout << str;
    }
  }
  rt_buf->msg_count = 0;
  device->unmap(runtime_buf);
}

bool CompiledTaichiKernel::check_ext_arr_read(int i) const {
  auto iter = ext_arr_access.find(i);
  if (iter == ext_arr_access.end())
    return false;

  return (iter->second & irpass::ExternalPtrAccess::READ) !=
         irpass::ExternalPtrAccess::NONE;
}

bool CompiledTaichiKernel::check_ext_arr_write(int i) const {
  auto iter = ext_arr_access.find(i);
  if (iter == ext_arr_access.end())
    return false;

  return (iter->second & irpass::ExternalPtrAccess::WRITE) !=
         irpass::ExternalPtrAccess::NONE;
}

void CompiledTaichiKernel::set_used(const UsedFeature &used) {
  this->used = used;
}

OpenGlRuntime::~OpenGlRuntime() {
  saved_arg_bufs.clear();
  impl.reset(nullptr);
  device.reset();
}

void DeviceCompiledTaichiKernel::launch(RuntimeContext &ctx,
                                        Kernel *kernel,
                                        OpenGlRuntime *runtime) const {
  uint8_t *args_buf_mapped = nullptr;
  auto args = kernel->args;
  // If we have external array args we'll have to do host-device memcpy.
  // Whether we get external array arg is runtime information.
  bool has_ext_arr = false;
  bool synced = false;

  if (program_.args_buf_size || program_.ret_buf_size) {
    args_buf_ =
        device_->allocate_memory_unique({taichi_opengl_external_arr_base,
                                         /*host_write=*/true,
                                         /*host_read=*/true,
                                         /*export_sharing=*/false});
  }

  // Prepare external arrays/ndarrays
  // - ctx.args[i] contains its ptr on host, it could be a raw ptr or
  // DeviceAllocation*
  // - For raw ptrs, its content will be synced to device through
  // ext_arr_bufs_[i] which is its corresponding DeviceAllocation on device.
  // Note shapes of these external arrays still reside in argument buffer,
  // see more details below.
  for (auto &item : program_.arr_args) {
    int i = item.first;
    TI_ASSERT(args[i].is_array);
    const auto arr_sz = ctx.array_runtime_sizes[i];
    if (arr_sz == 0 || ctx.is_device_allocations[i]) {
      continue;
    }
    has_ext_arr = true;
    if (arr_sz != item.second.runtime_size ||
        ext_arr_bufs_[i] == kDeviceNullAllocation) {
      if (ext_arr_bufs_[i] != kDeviceNullAllocation) {
        device_->dealloc_memory(ext_arr_bufs_[i]);
      }
      ext_arr_bufs_[i] = device_->allocate_memory({arr_sz, /*host_write=*/true,
                                                   /*host_read=*/true,
                                                   /*export_sharing=*/false});
      item.second.runtime_size = arr_sz;
    }
    void *host_ptr = (void *)ctx.args[i];
    void *baseptr = device_->map(ext_arr_bufs_[i]);
    if (program_.check_ext_arr_read(i)) {
      std::memcpy((char *)baseptr, host_ptr, arr_sz);
    }
    device_->unmap(ext_arr_bufs_[i]);
  }
  // clang-format off
  // Prepare argument buffer
  // Layout:
  // |              args               |  shape of ext arr  |  ret |
  // baseptr
  // |..taichi_opengl_extra_args_base..|
  // |...............taichi_opengl_ret_base.................|
  // |................taichi_opengl_external_arr_base..............|
  // clang-format on
  if (program_.args_buf_size) {
    args_buf_mapped = (uint8_t *)device_->map(*args_buf_);
    std::memcpy(args_buf_mapped, ctx.args,
                program_.arg_count * sizeof(uint64_t));
    if (program_.arr_args.size()) {
      std::memcpy(
          args_buf_mapped + size_t(taichi_opengl_extra_args_base),
          ctx.extra_args,
          size_t(program_.arg_count * taichi_max_num_indices) * sizeof(int));
    }
    device_->unmap(*args_buf_);
  }

  // Prepare runtime
  if (program_.used.print) {
    // TODO(archibate): use result_buffer for print results
    auto runtime_buf = runtime->impl->core_bufs.runtime;
    auto mapped = (GLSLRuntime *)device_->map(runtime_buf);
    mapped->msg_count = 0;
    device_->unmap(runtime_buf);
  }

  auto cmdlist = device_->get_compute_stream()->new_command_list();

  // Kernel dispatch
  int i = 0;
  for (const auto &task : program_.tasks) {
    auto binder = compiled_pipeline_[i]->resource_binder();
    auto &core_bufs = runtime->impl->core_bufs;
    binder->rw_buffer(0, static_cast<int>(GLBufId::Runtime), core_bufs.runtime);
    if (program_.used.buf_data)
      binder->rw_buffer(0, static_cast<int>(GLBufId::Root), core_bufs.root);
    binder->rw_buffer(0, static_cast<int>(GLBufId::Gtmp), core_bufs.gtmp);
    if (program_.args_buf_size || program_.ret_buf_size)
      binder->rw_buffer(0, static_cast<int>(GLBufId::Args), *args_buf_);
    // TODO: properly assert and throw if we bind more than allowed SSBOs.
    //       On most devices this number is 8. But I need to look up how
    //       to query this information so currently this is thrown from OpenGl.
    for (const auto [arg_id, bind_id] : program_.used.arr_arg_to_bind_idx) {
      if (ctx.is_device_allocations[arg_id]) {
        DeviceAllocation *ptr =
            static_cast<DeviceAllocation *>((void *)ctx.args[arg_id]);

        binder->rw_buffer(0, bind_id, *ptr);
      } else {
        binder->rw_buffer(0, bind_id, ext_arr_bufs_[arg_id]);
      }
    }

    cmdlist->bind_pipeline(compiled_pipeline_[i].get());
    if (i == 0)
      cmdlist->bind_resources(binder);
    cmdlist->dispatch(task.num_groups, 1, 1);
    cmdlist->memory_barrier();
    i++;
  }

  if (program_.used.print || has_ext_arr || program_.ret_buf_size) {
    // We'll do device->host memcpy later so sync is required.
    device_->get_compute_stream()->submit_synced(cmdlist.get());
    synced = true;
  } else {
    device_->get_compute_stream()->submit(cmdlist.get());
  }

  // Data read-back
  if (program_.used.print) {
    dump_message_buffer(device_, runtime->impl->core_bufs.runtime,
                        program_.str_table);
  }

  if (has_ext_arr) {
    for (auto &item : program_.arr_args) {
      int i = item.first;
      const auto arr_sz = ctx.array_runtime_sizes[i];
      if (arr_sz > 0 && !ctx.is_device_allocations[i]) {
        uint8_t *baseptr = (uint8_t *)device_->map(ext_arr_bufs_[i]);
        memcpy((void *)ctx.args[i], baseptr, arr_sz);
        device_->unmap(ext_arr_bufs_[i]);
      }
    }
  }

  if (program_.ret_buf_size) {
    uint8_t *baseptr = (uint8_t *)device_->map(*args_buf_);
    memcpy(runtime->result_buffer, baseptr + taichi_opengl_ret_base,
           program_.ret_buf_size);
    device_->unmap(*args_buf_);
  }
  if (program_.args_buf_size || program_.ret_buf_size) {
    runtime->saved_arg_bufs.push_back(std::move(args_buf_));
  }

  if (synced) {
    runtime->saved_arg_bufs.clear();
  }
}

DeviceCompiledTaichiKernel::DeviceCompiledTaichiKernel(
    CompiledTaichiKernel &&program,
    Device *device)
    : device_(device), program_(std::move(program)) {
  for (auto &t : program_.tasks) {
    compiled_pipeline_.push_back(device->create_pipeline(
        {PipelineSourceType::glsl_src, t.src.data(), t.src.size()}, t.name));
  }
}

OpenGlRuntime::OpenGlRuntime() {
  initialize_opengl();

  device = std::make_unique<GLDevice>();

  impl = std::make_unique<OpenGlRuntimeImpl>();

  impl->runtime = std::make_unique<GLSLRuntime>();
  impl->core_bufs.runtime = device->allocate_memory(
      {sizeof(GLSLRuntime), /*host_write=*/false, /*host_read=*/true});

  impl->core_bufs.gtmp =
      device->allocate_memory({taichi_global_tmp_buffer_size});

  auto cmdlist = device->get_compute_stream()->new_command_list();
  cmdlist->buffer_fill(impl->core_bufs.runtime.get_ptr(0), sizeof(GLSLRuntime),
                       0);
  cmdlist->buffer_fill(impl->core_bufs.gtmp.get_ptr(0),
                       taichi_global_tmp_buffer_size, 0);
  device->get_compute_stream()->submit_synced(cmdlist.get());
}

DeviceCompiledTaichiKernel *OpenGlRuntime::keep(
    CompiledTaichiKernel &&program) {
  auto p = std::make_unique<DeviceCompiledTaichiKernel>(std::move(program),
                                                        device.get());
  auto ptr = p.get();
  impl->programs.push_back(std::move(p));
  return ptr;
}

void OpenGlRuntime::add_snode_tree(size_t size) {
  impl->core_bufs.root = device->allocate_memory({size});

  auto cmdlist = device->get_compute_stream()->new_command_list();
  cmdlist->buffer_fill(impl->core_bufs.root.get_ptr(0), size, 0);
  device->get_compute_stream()->submit_synced(cmdlist.get());
}

bool is_opengl_api_available(bool use_gles) {
  if (get_environ_config("TI_ENABLE_OPENGL", 1) == 0)
    return false;
  return initialize_opengl(use_gles, true);
}

#else

struct OpenGlRuntimeImpl {};

OpenGlRuntime::OpenGlRuntime() {
  TI_NOT_IMPLEMENTED;
}

OpenGlRuntime::~OpenGlRuntime() {
  TI_NOT_IMPLEMENTED;
}

DeviceCompiledTaichiKernel *OpenGlRuntime::keep(
    CompiledTaichiKernel &&program) {
  TI_NOT_IMPLEMENTED;
  return nullptr;
}

void OpenGlRuntime::add_snode_tree(size_t size) {
  TI_NOT_IMPLEMENTED;
}

bool is_opengl_api_available(bool use_gles) {
  return false;
}

bool initialize_opengl(bool use_gles, bool error_tolerance) {
  TI_NOT_IMPLEMENTED;
}

#endif  // TI_WITH_OPENGL

bool is_gles() {
  return kUseGles;
}

}  // namespace opengl
}  // namespace lang
}  // namespace taichi
