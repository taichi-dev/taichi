//#define _GLSL_DEBUG 1
#include "opengl_api.h"

#include "taichi/backends/opengl/opengl_kernel_util.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/program/py_print_buffer.h"
#include "taichi/util/environ_config.h"
#include "taichi/backends/opengl/shaders/runtime.h"
#include "taichi/backends/opengl/shaders/listman.h"
#include "taichi/ir/transforms.h"

#ifdef TI_WITH_OPENGL
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "taichi/backends/opengl/opengl_device.h"
#endif

#include <list>

TLANG_NAMESPACE_BEGIN
namespace opengl {

#define PER_OPENGL_EXTENSION(x) bool opengl_extension_##x;
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION

// will later be initialized in initialize_opengl, here we use the minimum
// value according to OpenGL spec in case glGetIntegerv didn't work properly
int opengl_max_block_dim = 1024;
int opengl_max_grid_dim = 1024;

#ifdef TI_WITH_OPENGL

static std::string add_line_markers(std::string x) {
  std::string marker;
  size_t pos = 0, npos;
  int line = 0;
  while (1) {
    npos = x.find_first_of('\n', pos);
    marker = fmt::format("{:3d} ", ++line);
    if (npos == std::string::npos)
      break;
    x.insert(pos, marker);
    pos = npos + 1 + marker.size();
  }
  return x;
}

struct GLSLLauncherImpl {
  std::unique_ptr<Device> device;

  struct {
    DeviceAllocation runtime = kDeviceNullAllocation;
    DeviceAllocation listman = kDeviceNullAllocation;
    DeviceAllocation root = kDeviceNullAllocation;
    DeviceAllocation gtmp = kDeviceNullAllocation;
  } core_bufs;

  GLSLLauncherImpl() {
  }

  std::unique_ptr<GLSLRuntime> runtime;
  std::unique_ptr<GLSLListman> listman;

  std::vector<std::unique_ptr<CompiledProgram>> programs;
};

bool initialize_opengl(bool error_tolerance) {
  static std::optional<bool> supported;  // std::nullopt

  TI_TRACE("initialize_opengl({}) called", error_tolerance);

  if (supported.has_value()) {  // this function has been called before
    if (supported.value()) {    // detected to be true in last call
      return true;
    } else {
      if (!error_tolerance)  // not called from with_opengl
        TI_ERROR("OpenGL not supported");
      return false;
    }
  }

  glfwInit();
  // Compute Shader requires OpenGL 4.3+ (or OpenGL ES 3.1+)
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);
  // GL context needs a window (There's no true headless GL)
  GLFWwindow *window =
      glfwCreateWindow(1, 1, "Make OpenGL Context", nullptr, nullptr);
  if (!window) {
    const char *desc = nullptr;
    int status = glfwGetError(&desc);
    if (!desc)
      desc = "Unknown Error";
    if (error_tolerance) {
      // error tolerated, returning false
      TI_DEBUG("[glsl] cannot create GLFW window: error {}: {}", status, desc);
      supported = std::make_optional<bool>(false);
      return false;
    }
    TI_ERROR("[glsl] cannot create GLFW window: error {}: {}", status, desc);
  }
  glfwMakeContextCurrent(window);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    if (error_tolerance) {
      TI_WARN("[glsl] cannot initialize GLAD");
      supported = std::make_optional<bool>(false);
      return false;
    }
    TI_ERROR("[glsl] cannot initialize GLAD");
  }
#define PER_OPENGL_EXTENSION(x)          \
  if ((opengl_extension_##x = GLAD_##x)) \
    TI_TRACE("[glsl] Found " #x);
#include "taichi/inc/opengl_extension.inc.h"
#undef PER_OPENGL_EXTENSION
  if (!opengl_extension_GL_ARB_compute_shader) {
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
  return true;
}

struct CompiledKernel {
  std::string kernel_name;
  std::unique_ptr<Pipeline> pipeline;
  int workgroup_size;
  int num_groups;

  CompiledKernel(const std::string &kernel_name_,
                 const std::string &kernel_source_code,
                 Device *device,
                 int _workgroup_size,
                 int _num_groups)
      : kernel_name(kernel_name_) {
    num_groups = std::min(_num_groups, opengl_max_grid_dim);
    workgroup_size = std::min(_workgroup_size, opengl_max_block_dim);

    size_t layout_pos = kernel_source_code.find("precision highp float;\n");
    TI_ASSERT(layout_pos != std::string::npos);
    std::string source =
        kernel_source_code.substr(0, layout_pos) +
        fmt::format(
            "layout(local_size_x = {}, local_size_y = 1, local_size_z = "
            "1) in;\n",
            workgroup_size) +
        kernel_source_code.substr(layout_pos);

    TI_TRACE("[glsl]\ncompiling kernel {}<<<{}, {}>>>:\n{}", kernel_name,
             num_groups, workgroup_size, kernel_source_code);

    pipeline = device->create_pipeline(
        {PipelineSourceType::glsl_src, source.data(), source.size()},
        kernel_name);
  }
};

struct CompiledProgram::Impl {
  std::vector<std::unique_ptr<CompiledKernel>> kernels;

  int arg_count{0};
  int ret_count{0};
  size_t args_buf_size{0};
  size_t total_ext_arr_size{0};
  size_t ret_buf_size{0};

  std::unordered_map<int, size_t> ext_arr_map;
  std::unordered_map<int, irpass::ExternalPtrAccess> ext_arr_access;
  std::vector<std::string> str_table;
  UsedFeature used;
  Device *device;

  DeviceAllocation args_buf{kDeviceNullAllocation};
  DeviceAllocation ext_arr_buf{kDeviceNullAllocation};
  DeviceAllocation ret_buf{kDeviceNullAllocation};

  Impl(Kernel *kernel, Device *device) : device(device) {
    arg_count = kernel->args.size();
    ret_count = kernel->rets.size();
    for (int i = 0; i < arg_count; i++) {
      if (kernel->args[i].is_external_array) {
        ext_arr_map[i] = kernel->args[i].size;
      }
    }

    for (const auto &[i, size] : ext_arr_map) {
      total_ext_arr_size += size;
    }

    args_buf_size = arg_count * sizeof(uint64_t);
    if (ext_arr_map.size()) {
      args_buf_size = taichi_opengl_earg_base +
                      arg_count * taichi_max_num_indices * sizeof(int);
    }

    ret_buf_size = ret_count * sizeof(uint64_t);

    if (args_buf_size) {
      args_buf = device->allocate_memory({args_buf_size, /*host_write=*/true,
                                          /*host_read=*/false,
                                          /*export_sharing=*/false});
    }

    if (total_ext_arr_size) {
      // Set both host write & host read for now
      ext_arr_buf =
          device->allocate_memory({total_ext_arr_size, /*host_write=*/true,
                                   /*host_read=*/true,
                                   /*export_sharing=*/false});
    }

    if (ret_buf_size) {
      ret_buf = device->allocate_memory({ret_buf_size, /*host_write=*/false,
                                         /*host_read=*/true,
                                         /*export_sharing=*/false});
    }
  }

  void add(const std::string &kernel_name,
           const std::string &kernel_source_code,
           int num_workgrpus,
           int workgroup_size,
           std::unordered_map<int, irpass::ExternalPtrAccess> *ext_ptr_access) {
    kernels.push_back(std::make_unique<CompiledKernel>(
        kernel_name, kernel_source_code, device, workgroup_size,
        num_workgrpus));
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

  int lookup_or_add_string(const std::string &str) {
    int i;
    for (i = 0; i < str_table.size(); i++) {
      if (str_table[i] == str) {
        return i;
      }
    }
    str_table.push_back(str);
    return i;
  }

  void dump_message_buffer(GLSLLauncher *launcher) const {
    auto runtime = launcher->impl->core_bufs.runtime;
    auto rt_buf = (GLSLRuntime *)device->map(launcher->impl->core_bufs.runtime);

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
    device->unmap(launcher->impl->core_bufs.runtime);
  }

  bool check_ext_arr_read(int i) const {
    auto iter = ext_arr_access.find(i);
    if (iter == ext_arr_access.end())
      return false;

    return (iter->second & irpass::ExternalPtrAccess::READ) !=
           irpass::ExternalPtrAccess::NONE;
  }

  bool check_ext_arr_write(int i) const {
    auto iter = ext_arr_access.find(i);
    if (iter == ext_arr_access.end())
      return false;

    return (iter->second & irpass::ExternalPtrAccess::WRITE) !=
           irpass::ExternalPtrAccess::NONE;
  }

  GLbitfield get_ext_arr_access(size_t &total_ext_arr_size) const {
    GLbitfield access = 0;
    for (const auto &[i, size] : ext_arr_map) {
      total_ext_arr_size += size;
      if (check_ext_arr_read(i)) {
        access |= GL_MAP_WRITE_BIT;
      }
      if (check_ext_arr_write(i)) {
        access |= GL_MAP_READ_BIT;
      }
    }
    return access;
  }

  void launch(Context &ctx, GLSLLauncher *launcher) const {
    std::array<void *, taichi_max_num_args> ext_arr_host_ptrs;

    uint8_t *args_buf_mapped = nullptr;

    // Prepare external array
    if (total_ext_arr_size) {
      void *baseptr = device->map(ext_arr_buf);

      size_t accum_size = 0;
      for (const auto &[i, size] : ext_arr_map) {
        auto ptr = (void *)ctx.args[i];
        ctx.args[i] = accum_size;
        ext_arr_host_ptrs[i] = ptr;
        if (check_ext_arr_read(i)) {
          std::memcpy((char *)baseptr + accum_size, ptr, size);
        }
        accum_size += size;
      }

      device->unmap(ext_arr_buf);
    }

    // Prepare argument buffer
    if (args_buf_size) {
      args_buf_mapped = (uint8_t *)device->map(args_buf);
      std::memcpy(args_buf_mapped, ctx.args, arg_count * sizeof(uint64_t));
      if (ext_arr_map.size()) {
        std::memcpy(args_buf_mapped + size_t(taichi_opengl_earg_base),
                    ctx.extra_args,
                    size_t(arg_count * taichi_max_num_indices) * sizeof(int));
      }
      device->unmap(args_buf);
    }

    // Prepare runtime
    if (used.print) {
      // TODO(archibate): use result_buffer for print results
      auto runtime_buf = launcher->impl->core_bufs.runtime;
      auto mapped = (GLSLRuntime *)device->map(runtime_buf);
      mapped->msg_count = 0;
      device->unmap(runtime_buf);
    }

    auto cmdlist = device->get_compute_stream()->new_command_list();

    // Kernel dispatch
    for (const auto &kernel : kernels) {
      auto binder = kernel->pipeline->resource_binder();
      auto &core_bufs = launcher->impl->core_bufs;
      binder->buffer(0, int(GLBufId::Runtime), core_bufs.runtime);
      binder->buffer(0, int(GLBufId::Listman), core_bufs.listman);
      binder->buffer(0, int(GLBufId::Root), core_bufs.root);
      binder->buffer(0, int(GLBufId::Gtmp), core_bufs.gtmp);
      if (args_buf_size)
        binder->buffer(0, int(GLBufId::Args), args_buf);
      if (ret_buf_size)
        binder->buffer(0, int(GLBufId::Retr), ret_buf);
      if (total_ext_arr_size)
        binder->buffer(0, int(GLBufId::Extr), ext_arr_buf);

      cmdlist->bind_pipeline(kernel->pipeline.get());
      cmdlist->bind_resources(binder);
      cmdlist->dispatch(kernel->num_groups, 1, 1);
      cmdlist->memory_barrier();
    }

    if (used.print || total_ext_arr_size || ret_buf_size) {
      device->get_compute_stream()->submit_synced(cmdlist.get());
    } else {
      device->get_compute_stream()->submit(cmdlist.get());
    }

    // Data read-back
    if (used.print) {
      dump_message_buffer(launcher);
    }

    if (total_ext_arr_size) {
      uint8_t *baseptr = (uint8_t *)device->map(ext_arr_buf);
      for (const auto &[i, size] : ext_arr_map) {
        memcpy(ext_arr_host_ptrs[i], baseptr + size_t(ctx.args[i]), size);
      }
      device->unmap(ext_arr_buf);
    }

    if (ret_buf_size) {
      memcpy(launcher->result_buffer, device->map(ret_buf), ret_buf_size);
      device->unmap(ret_buf);
    }
  }
};

GLSLLauncher::GLSLLauncher(size_t root_size) {
  initialize_opengl();

  device = std::make_unique<GLDevice>();

  impl = std::make_unique<GLSLLauncherImpl>();

  impl->runtime = std::make_unique<GLSLRuntime>();
  impl->core_bufs.runtime = device->allocate_memory(
      {sizeof(GLSLRuntime), /*host_write=*/false, /*host_read=*/true});

  impl->listman = std::make_unique<GLSLListman>();
  impl->core_bufs.listman = device->allocate_memory({sizeof(GLSLListman)});

  impl->core_bufs.root = device->allocate_memory({root_size});

  impl->core_bufs.gtmp =
      device->allocate_memory({taichi_global_tmp_buffer_size});

  auto cmdlist = device->get_compute_stream()->new_command_list();
  cmdlist->buffer_fill(impl->core_bufs.runtime.get_ptr(0), sizeof(GLSLRuntime),
                       0);
  cmdlist->buffer_fill(impl->core_bufs.listman.get_ptr(0), sizeof(GLSLListman),
                       0);
  cmdlist->buffer_fill(impl->core_bufs.root.get_ptr(0), root_size, 0);
  cmdlist->buffer_fill(impl->core_bufs.gtmp.get_ptr(0),
                       taichi_global_tmp_buffer_size, 0);
  device->get_compute_stream()->submit_synced(cmdlist.get());
}

void GLSLLauncher::keep(std::unique_ptr<CompiledProgram> program) {
  impl->programs.push_back(std::move(program));
}

bool is_opengl_api_available() {
  if (get_environ_config("TI_ENABLE_OPENGL", 1) == 0)
    return false;
  return initialize_opengl(true);
}

#else
struct GLProgram {};
struct GLSLLauncherImpl {};

struct CompiledProgram::Impl {
  UsedFeature used;

  Impl(Kernel *kernel, Device *device) {
    TI_NOT_IMPLEMENTED;
  }

  void add(const std::string &kernel_name,
           const std::string &kernel_source_code,
           int num_workgrpus,
           int workgroup_size,
           std::unordered_map<int, irpass::ExternalPtrAccess> *ext_ptr_access) {
    TI_NOT_IMPLEMENTED;
  }

  int lookup_or_add_string(const std::string &str) {
    TI_NOT_IMPLEMENTED;
  }

  void launch(Context &ctx, GLSLLauncher *launcher) const {
    TI_NOT_IMPLEMENTED;
  }
};

GLSLLauncher::GLSLLauncher(size_t size) {
  TI_NOT_IMPLEMENTED;
}

void GLSLLauncher::keep(std::unique_ptr<CompiledProgram>) {
  TI_NOT_IMPLEMENTED;
}

bool is_opengl_api_available() {
  return false;
}

bool initialize_opengl(bool error_tolerance) {
  TI_NOT_IMPLEMENTED;
}

#endif  // TI_WITH_OPENGL

CompiledProgram::CompiledProgram(Kernel *kernel, Device *device)
    : impl(std::make_unique<Impl>(kernel, device)) {
}

CompiledProgram::~CompiledProgram() = default;

void CompiledProgram::add(
    const std::string &kernel_name,
    const std::string &kernel_source_code,
    int num_workgrous,
    int workgroup_size,
    std::unordered_map<int, irpass::ExternalPtrAccess> *ext_ptr_access) {
  impl->add(kernel_name, kernel_source_code, num_workgrous, workgroup_size,
            ext_ptr_access);
}

void CompiledProgram::set_used(const UsedFeature &used) {
  impl->used = used;
}

int CompiledProgram::lookup_or_add_string(const std::string &str) {
  return impl->lookup_or_add_string(str);
}

void CompiledProgram::launch(Context &ctx, GLSLLauncher *launcher) const {
  impl->launch(ctx, launcher);
}

GLSLLauncher::~GLSLLauncher() = default;

}  // namespace opengl
TLANG_NAMESPACE_END
