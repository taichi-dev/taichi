//#define _GLSL_DEBUG 1
#include "opengl_api.h"

#include "taichi/backends/opengl/opengl_kernel_util.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/program/py_print_buffer.h"
#include "taichi/util/environ_config.h"
#include "taichi/backends/opengl/shaders/runtime.h"
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

struct OpenGlRuntimeImpl {
  struct {
    DeviceAllocation runtime = kDeviceNullAllocation;
    DeviceAllocation root = kDeviceNullAllocation;
    DeviceAllocation gtmp = kDeviceNullAllocation;
  } core_bufs;

  OpenGlRuntimeImpl() {
  }

  std::unique_ptr<GLSLRuntime> runtime{nullptr};
  std::vector<std::unique_ptr<DeviceCompiledProgram>> programs;
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

void CompiledProgram::init_args(Kernel *kernel) {
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
}

void CompiledProgram::add(
    const std::string &kernel_name,
    const std::string &kernel_source_code,
    int num_workgroups,
    int workgroup_size,
    std::unordered_map<int, irpass::ExternalPtrAccess> *ext_ptr_access) {
  num_workgroups = std::min(num_workgroups, opengl_max_grid_dim);
  workgroup_size = std::min(workgroup_size, opengl_max_block_dim);

  size_t layout_pos = kernel_source_code.find("precision highp float;\n");
  TI_ASSERT(layout_pos != std::string::npos);
  std::string source =
      kernel_source_code.substr(0, layout_pos) +
      fmt::format(
          "layout(local_size_x = {}, local_size_y = 1, local_size_z = "
          "1) in;\n",
          workgroup_size) +
      kernel_source_code.substr(layout_pos);

  TI_DEBUG("[glsl]\ncompiling kernel {}<<<{}, {}>>>:\n{}", kernel_name,
           num_workgroups, workgroup_size, source);
  kernels.push_back({kernel_name, source, workgroup_size, num_workgroups});

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

int CompiledProgram::lookup_or_add_string(const std::string &str) {
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

bool CompiledProgram::check_ext_arr_read(int i) const {
  auto iter = ext_arr_access.find(i);
  if (iter == ext_arr_access.end())
    return false;

  return (iter->second & irpass::ExternalPtrAccess::READ) !=
         irpass::ExternalPtrAccess::NONE;
}

bool CompiledProgram::check_ext_arr_write(int i) const {
  auto iter = ext_arr_access.find(i);
  if (iter == ext_arr_access.end())
    return false;

  return (iter->second & irpass::ExternalPtrAccess::WRITE) !=
         irpass::ExternalPtrAccess::NONE;
}

void CompiledProgram::set_used(const UsedFeature &used) {
  this->used = used;
}

OpenGlRuntime::~OpenGlRuntime() = default;

void DeviceCompiledProgram::launch(Context &ctx, OpenGlRuntime *runtime) const {
  std::array<void *, taichi_max_num_args> ext_arr_host_ptrs;

  uint8_t *args_buf_mapped = nullptr;

  // Prepare external array: copy from ctx.args[i] (which is a host pointer
  // pointing to the external array) to device, and save the accumulated copied
  // size information. Note here we copy external array to Arg buffer in
  // runtime. Its layout is shown below:
  // |           args            |    shape of ext arr        |   ext arr   |
  // baseptr
  // |..taichi_opengl_earg_base..|
  // |.................ext_arr_offset.........................|
  // |.......................ctx.args[i]............................|
  //                                                     i-th arg (ext arr)
  // We save each external array's offset from args_buf_ baseptr back to
  // ctx.args[i].
  if (program_.total_ext_arr_size) {
    void *baseptr = device_->map(args_buf_);
    size_t ext_arr_offset =
        size_t(taichi_opengl_earg_base) +
        sizeof(int) * size_t(program_.arg_count * taichi_max_num_indices);
    size_t accum_size = 0;
    for (const auto &[i, size] : program_.ext_arr_map) {
      auto ptr = (void *)ctx.args[i];
      ctx.args[i] = accum_size + ext_arr_offset;
      ext_arr_host_ptrs[i] = ptr;
      if (program_.check_ext_arr_read(i)) {
        std::memcpy((char *)baseptr + ctx.args[i], ptr, size);
      }
      accum_size += size;
    }

    device_->unmap(args_buf_);
  }

  // Prepare argument buffer
  if (program_.args_buf_size) {
    args_buf_mapped = (uint8_t *)device_->map(args_buf_);
    std::memcpy(args_buf_mapped, ctx.args,
                program_.arg_count * sizeof(uint64_t));
    if (program_.ext_arr_map.size()) {
      std::memcpy(
          args_buf_mapped + size_t(taichi_opengl_earg_base), ctx.extra_args,
          size_t(program_.arg_count * taichi_max_num_indices) * sizeof(int));
    }
    device_->unmap(args_buf_);
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
  for (const auto &kernel : program_.kernels) {
    auto binder = compiled_pipeline_[i]->resource_binder();
    auto &core_bufs = runtime->impl->core_bufs;
    binder->buffer(0, int(GLBufId::Runtime), core_bufs.runtime);
    binder->buffer(0, int(GLBufId::Root), core_bufs.root);
    binder->buffer(0, int(GLBufId::Gtmp), core_bufs.gtmp);
    if (program_.args_buf_size)
      binder->buffer(0, int(GLBufId::Args), args_buf_);
    if (program_.ret_buf_size)
      binder->buffer(0, int(GLBufId::Retr), ret_buf_);

    cmdlist->bind_pipeline(compiled_pipeline_[i].get());
    cmdlist->bind_resources(binder);
    cmdlist->dispatch(kernel.num_groups, 1, 1);
    cmdlist->memory_barrier();
    i++;
  }

  if (program_.used.print || program_.total_ext_arr_size ||
      program_.ret_buf_size) {
    device_->get_compute_stream()->submit_synced(cmdlist.get());
  } else {
    device_->get_compute_stream()->submit(cmdlist.get());
  }

  // Data read-back
  if (program_.used.print) {
    dump_message_buffer(device_, runtime->impl->core_bufs.runtime,
                        program_.str_table);
  }

  if (program_.total_ext_arr_size) {
    uint8_t *baseptr = (uint8_t *)device_->map(args_buf_);
    for (const auto &[i, size] : program_.ext_arr_map) {
      memcpy(ext_arr_host_ptrs[i], baseptr + size_t(ctx.args[i]), size);
    }
    device_->unmap(args_buf_);
  }

  if (program_.ret_buf_size) {
    memcpy(runtime->result_buffer, device_->map(ret_buf_),
           program_.ret_buf_size);
    device_->unmap(ret_buf_);
  }
}

DeviceCompiledProgram::DeviceCompiledProgram(CompiledProgram &&program,
                                             Device *device)
    : program_(std::move(program)), device_(device) {
  if (program_.args_buf_size || program_.total_ext_arr_size) {
    args_buf_ = device->allocate_memory(
        {program_.args_buf_size + program_.total_ext_arr_size,
         /*host_write=*/true,
         /*host_read=*/true,
         /*export_sharing=*/false});
  }

  if (program_.ret_buf_size) {
    ret_buf_ =
        device->allocate_memory({program_.ret_buf_size, /*host_write=*/false,
                                 /*host_read=*/true,
                                 /*export_sharing=*/false});
  }

  for (auto &k : program_.kernels) {
    compiled_pipeline_.push_back(
        device->create_pipeline({PipelineSourceType::glsl_src,
                                 k.kernel_src.data(), k.kernel_src.size()},
                                k.kernel_name));
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

DeviceCompiledProgram *OpenGlRuntime::keep(CompiledProgram &&program) {
  auto p =
      std::make_unique<DeviceCompiledProgram>(std::move(program), device.get());
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

bool is_opengl_api_available() {
  if (get_environ_config("TI_ENABLE_OPENGL", 1) == 0)
    return false;
  return initialize_opengl(true);
}

#else

struct OpenGlRuntimeImpl {};

OpenGlRuntime::OpenGlRuntime() {
  TI_NOT_IMPLEMENTED;
}

OpenGlRuntime::~OpenGlRuntime() {
  TI_NOT_IMPLEMENTED;
}

DeviceCompiledProgram *OpenGlRuntime::keep(CompiledProgram &&program) {
  TI_NOT_IMPLEMENTED;
  return nullptr;
}

void OpenGlRuntime::add_snode_tree(size_t size) {
  TI_NOT_IMPLEMENTED;
}

bool is_opengl_api_available() {
  return false;
}

bool initialize_opengl(bool error_tolerance) {
  TI_NOT_IMPLEMENTED;
}

#endif  // TI_WITH_OPENGL

}  // namespace opengl
TLANG_NAMESPACE_END
