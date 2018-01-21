/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#ifdef TC_USE_OPENGL

#include <taichi/system/opengl.h>
#include <taichi/system/timer.h>
#include <taichi/common/util.h>

TC_NAMESPACE_BEGIN

bool GLWindow::pressed[256]{false};
int GLWindow::x_start_position = 0;

GLuint load_shader(GLint type, string filename) {
#ifdef _WIN64
  FILE *file;
  fopen_s(&file, filename.c_str(), "r");
#else
  FILE *file = fopen(filename.c_str(), "r");
#endif
  if (!file) {
    printf("Error: shader %s not found!\n", filename.c_str());
    assert(false);
    exit(-1);
  }
  char source[10000];
  char *p_source = source;
  unsigned len = (unsigned)fread(source, 1, 10000, file);
  source[len] = 0;
  GLint shader = glCreateShader(type);
  glShaderSource(shader, 1, (const GLchar **)&p_source, NULL);
  glCompileShader(shader);
  GLint shader_compile_result;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &shader_compile_result);
  if (!shader_compile_result) {
    printf("Shader file %s failed to compile!\n", filename.c_str());
    int log_length;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    char *log = new char[log_length + 1];
    glGetShaderInfoLog(shader, log_length, nullptr, log);
    printf("****** Compile Info:\n");
    printf("%s\n", log);
    delete[] log;
    assert(false);
  }
  printf("Shader %s Compiled successfully.\n", filename.c_str());
  return shader;
}

GLuint load_program(string vertex_fn, string fragment_fn) {
  GLint v_shader =
            load_shader(GL_VERTEX_SHADER, "shaders/" + vertex_fn + ".vertex"),
        f_shader = load_shader(GL_FRAGMENT_SHADER,
                               "shaders/" + fragment_fn + ".fragment");
  GLint program = glCreateProgram();
  glAttachShader(program, v_shader);
  glAttachShader(program, f_shader);
  glLinkProgram(program);
  GLint link_result;
  glGetProgramiv(program, GL_LINK_STATUS, &link_result);
  if (!link_result) {
    printf("Linking error!\n");
    exit(-1);
  }
  return program;
}

void gl_print_error_code(int code) {
  switch (code) {
    case GL_INVALID_ENUM:
      printf("Invalid Enum");
      break;
    case GL_INVALID_VALUE:
      printf("Invalid Value");
      break;
    case GL_INVALID_OPERATION:
      printf("Invalid Operation");
      break;
    case GL_INVALID_FRAMEBUFFER_OPERATION:
      printf("Invalid Framebuffer Operation");
      break;
    case GL_OUT_OF_MEMORY:
      printf("Out of Memory");
      break;
    case GL_STACK_OVERFLOW:
      printf("Stack Overflow");
      break;
    case GL_STACK_UNDERFLOW:
      printf("Stack Underflow");
      break;
  }
  printf("\n");
}

bool GLWindow::first_window_created = false;
GLFWwindow *GLWindow::first_window = nullptr;
std::map<GLFWwindow *, GLWindow *> GLWindow::window_to_instance;

GLWindow::GLWindow(Config config) {
  int version = config.get("version", 41);
  width = config.get("width", 400);
  height = config.get("height", 400);
  title = config.get("title", "Tinge");
  background_color = config.get("background_color", Vector3(0));
  exit_on_esc = config.get("exit_on_esc", true);
  assert_info(glfwInit() == GL_TRUE, "Failed to initialize GLFW!");
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, version / 10);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, version % 10);
  if (version > 20) {
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  }

  window =
      glfwCreateWindow(width, height, title.c_str(), nullptr, first_window);
  glfwSetWindowPos(window, x_start_position + config.get("offset_x", 300),
                   config.get("offset_y", 300));
  x_start_position += width + 15;

  assert_info(window != nullptr, "Unable to create OpenGL context!\n");

  glfwMakeContextCurrent(window);
  // glfwSetKeyCallback(window, key_callback);

  glfwSwapInterval(1);
  // glfwSetCursorPosCallback(window, mouse_pos_callback);
  glewExperimental = true;
  assert_info(glewInit() == GLEW_OK, "Unable to initialize GLEW!");

  if (first_window_created == false) {
    first_window_created = true;
    first_window = window;
  }

  window_to_instance[window] = this;

  glfwSetCursorPosCallback(window, GLWindow::general_mouse_move_callback);
  glfwSetKeyCallback(window, GLWindow::general_keyboard_callback);
  glGetError();
  if (config.get("hide", false))
    glfwHideWindow(window);
}

GLWindow::~GLWindow() {
  window_to_instance.erase(window);
}

void GLWindow::print_actual_version() {
  printf("Actual OpenGL Version %s\n", glGetString(GL_VERSION));
  printf("Actual GLSL Version %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
}

void GLWindow::begin_rendering() {
  glfwMakeContextCurrent(window);
  glfwPollEvents();
  glClearColor(background_color.r, background_color.g, background_color.b, 1);
  glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
}

void GLWindow::keyboard_callback(int key, int scancode, int action, int mods) {
  if (action == GLFW_PRESS || action == GLFW_REPEAT)
    waiting_for_key = false;
  if (exit_on_esc && key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    exit(0);
  for (auto &callback : keyboard_callbacks) {
    callback(key, scancode, action, mods);
  }
}

RenderingGuard::RenderingGuard(GLWindow *target) {
  this->target = target;
  this->target->begin_rendering();
}

RenderingGuard::~RenderingGuard() {
  this->target->end_rendering();
}

ContextGuard::ContextGuard(GLWindow *target) {
  this->old_context = glfwGetCurrentContext();
  target->make_current();
}

ContextGuard::~ContextGuard() {
  glfwMakeContextCurrent(old_context);
}

TC_NAMESPACE_END

#endif
