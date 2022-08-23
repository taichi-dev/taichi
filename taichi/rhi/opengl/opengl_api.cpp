#include "opengl_api.h"

#include <list>

#include "glad/gl.h"
#include "glad/egl.h"
#include "GLFW/glfw3.h"
#include "taichi/rhi/opengl/opengl_device.h"

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

static void glfw_error_callback(int code, const char *description) {
  TI_WARN("GLFW Error {}: {}", code, description);
}

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
    glfwSetErrorCallback(glfw_error_callback);
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

bool is_opengl_api_available(bool use_gles) {
  return initialize_opengl(use_gles, true);
}

bool is_gles() {
  return kUseGles;
}

std::shared_ptr<Device> make_opengl_device() {
  std::shared_ptr<Device> dev = std::make_shared<GLDevice>();
  dev->set_cap(DeviceCapability::spirv_has_int64, true);
  dev->set_cap(DeviceCapability::spirv_has_float64, true);
  dev->set_cap(DeviceCapability::spirv_version, 0x10300);
  return dev;
}

}  // namespace opengl
}  // namespace lang
}  // namespace taichi
