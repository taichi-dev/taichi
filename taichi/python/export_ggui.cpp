
#include <vector>
#include "pybind11/pybind11.h"
#include <pybind11/numpy.h>
#include "pybind11/stl.h"

#include "taichi/common/interface.h"
#include "taichi/common/core.h"

namespace py = pybind11;

#ifdef TI_WITH_GGUI

#include "taichi/ui/utils/utils.h"
#include "taichi/ui/common/window_base.h"
#include "taichi/ui/backends/vulkan/window.h"
#include "taichi/ui/common/canvas_base.h"
#include "taichi/ui/common/camera.h"
#include "taichi/ui/backends/vulkan/canvas.h"
#include "taichi/ui/backends/vulkan/scene.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/arch.h"
#include "taichi/program/field_info.h"
#include "taichi/ui/common/gui_base.h"
#include "taichi/program/ndarray.h"
#include <memory>

TI_UI_NAMESPACE_BEGIN

using namespace taichi::lang;

glm::vec3 tuple_to_vec3(pybind11::tuple t) {
  return glm::vec3(t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>());
}

pybind11::tuple vec3_to_tuple(glm::vec3 v) {
  return pybind11::make_tuple(v.x, v.y, v.z);
}

// Here we convert the 2d-array to numpy array using pybind. Refs:
// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html?highlight=array_t#vectorizing-functions
// https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
py::array_t<float> mat4_to_nparray(glm::mat4 mat) {
  // Here we must explicitly pass args using py::detail::any_container<ssize_t>.
  // Refs:
  // https://stackoverflow.com/questions/54055530/error-no-matching-function-for-call-to-pybind11buffer-infobuffer-info
  return py::array_t<float>(
      py::detail::any_container<ssize_t>({4, 4}),  // shape (rows, cols)
      py::detail::any_container<ssize_t>(
          {sizeof(float) * 4, sizeof(float)}),  // strides in bytes
      glm::value_ptr(mat),                      // buffer pointer
      nullptr);
}

struct PyGui {
  GuiBase *gui;  // not owned
  void begin(std::string name, float x, float y, float width, float height) {
    gui->begin(name, x, y, width, height);
  }
  void end() {
    gui->end();
  }
  void text(std::string text) {
    gui->text(text);
  }
  bool checkbox(std::string name, bool old_value) {
    return gui->checkbox(name, old_value);
  }
  int slider_int(std::string name, int old_value, int minimum, int maximum) {
    return gui->slider_int(name, old_value, minimum, maximum);
  }
  float slider_float(std::string name,
                     float old_value,
                     float minimum,
                     float maximum) {
    return gui->slider_float(name, old_value, minimum, maximum);
  }
  py::tuple color_edit_3(std::string name, py::tuple old_value) {
    glm::vec3 old_color = tuple_to_vec3(old_value);
    glm::vec3 new_color = gui->color_edit_3(name, old_color);
    return vec3_to_tuple(new_color);
  }
  bool button(std::string name) {
    return gui->button(name);
  }
};

struct PyCamera {
  Camera camera;
  void position(float x, float y, float z) {
    camera.position = glm::vec3(x, y, z);
  }
  void lookat(float x, float y, float z) {
    camera.lookat = glm::vec3(x, y, z);
  }
  void up(float x, float y, float z) {
    camera.up = glm::vec3(x, y, z);
  }
  void projection_mode(ProjectionMode mode) {
    camera.projection_mode = mode;
  }
  void fov(float fov_) {
    camera.fov = fov_;
  }
  void left(float left_) {
    camera.left = left_;
  }
  void right(float right_) {
    camera.right = right_;
  }
  void top(float top_) {
    camera.top = top_;
  }
  void bottom(float bottom_) {
    camera.bottom = bottom_;
  }
  void z_near(float z_near_) {
    camera.z_near = z_near_;
  }
  void z_far(float z_far_) {
    camera.z_far = z_far_;
  }
  py::array_t<float> get_view_matrix() {
    return mat4_to_nparray(camera.get_view_matrix());
  }
  py::array_t<float> get_projection_matrix(float aspect_ratio) {
    return mat4_to_nparray(camera.get_projection_matrix(aspect_ratio));
  }
};

struct PyScene {
  SceneBase *scene;  // owned

  PyScene() {
    // todo: support other ggui backends
    scene = new vulkan::Scene();
  }

  void set_camera(PyCamera camera) {
    scene->set_camera(camera.camera);
  }

  void lines(FieldInfo vbo,
             FieldInfo indices,
             bool has_per_vertex_color,
             py::tuple color_,
             float width,
             float draw_index_count,
             float draw_first_index,
             float draw_vertex_count,
             float draw_first_vertex) {
    RenderableInfo renderable_info;
    renderable_info.vbo = vbo;
    renderable_info.indices = indices;
    renderable_info.has_per_vertex_color = has_per_vertex_color;
    renderable_info.has_user_customized_draw = true;
    renderable_info.draw_index_count = (int)draw_index_count;
    renderable_info.draw_first_index = (int)draw_first_index;
    renderable_info.draw_vertex_count = (int)draw_vertex_count;
    renderable_info.draw_first_vertex = (int)draw_first_vertex;

    SceneLinesInfo info;
    info.renderable_info = renderable_info;
    info.color = tuple_to_vec3(color_);
    info.width = width;

    return scene->lines(info);
  }

  void mesh(FieldInfo vbo,
            bool has_per_vertex_color,
            FieldInfo indices,
            py::tuple color,
            bool two_sided,
            float draw_index_count,
            float draw_first_index,
            float draw_vertex_count,
            float draw_first_vertex,
            bool show_wireframe) {
    RenderableInfo renderable_info;
    renderable_info.vbo = vbo;
    renderable_info.has_per_vertex_color = has_per_vertex_color;
    renderable_info.indices = indices;
    renderable_info.has_user_customized_draw = true;
    renderable_info.draw_index_count = (int)draw_index_count;
    renderable_info.draw_first_index = (int)draw_first_index;
    renderable_info.draw_vertex_count = (int)draw_vertex_count;
    renderable_info.draw_first_vertex = (int)draw_first_vertex;
    renderable_info.display_mode = show_wireframe
                                       ? taichi::lang::PolygonMode::Line
                                       : taichi::lang::PolygonMode::Fill;

    MeshInfo info;
    info.renderable_info = renderable_info;
    info.color = tuple_to_vec3(color);
    info.two_sided = two_sided;

    scene->mesh(info);
  }

  void particles(FieldInfo vbo,
                 bool has_per_vertex_color,
                 py::tuple color_,
                 float radius,
                 float draw_vertex_count,
                 float draw_first_vertex) {
    RenderableInfo renderable_info;
    renderable_info.vbo = vbo;
    renderable_info.has_user_customized_draw = true;
    renderable_info.has_per_vertex_color = has_per_vertex_color;
    renderable_info.draw_vertex_count = (int)draw_vertex_count;
    renderable_info.draw_first_vertex = (int)draw_first_vertex;

    ParticlesInfo info;
    info.renderable_info = renderable_info;
    info.color = tuple_to_vec3(color_);
    info.radius = radius;

    scene->particles(info);
  }

  void mesh_instance(FieldInfo vbo,
                     bool has_per_vertex_color,
                     FieldInfo indices,
                     py::tuple color,
                     bool two_sided,
                     FieldInfo transforms,
                     float draw_instance_count,
                     float draw_first_instance,
                     float draw_index_count,
                     float draw_first_index,
                     float draw_vertex_count,
                     float draw_first_vertex,
                     bool show_wireframe) {
    RenderableInfo renderable_info;
    renderable_info.vbo = vbo;
    renderable_info.has_per_vertex_color = has_per_vertex_color;
    renderable_info.indices = indices;
    renderable_info.has_user_customized_draw = true;
    renderable_info.draw_index_count = (int)draw_index_count;
    renderable_info.draw_first_index = (int)draw_first_index;
    renderable_info.draw_vertex_count = (int)draw_vertex_count;
    renderable_info.draw_first_vertex = (int)draw_first_vertex;
    renderable_info.display_mode = show_wireframe
                                       ? taichi::lang::PolygonMode::Line
                                       : taichi::lang::PolygonMode::Fill;

    MeshInfo info;
    info.renderable_info = renderable_info;
    info.color = tuple_to_vec3(color);
    info.two_sided = two_sided;
    if (transforms.valid) {
      info.start_instance = (int)draw_first_instance;
      info.num_instances =
          (draw_instance_count + info.start_instance) > transforms.shape[0]
              ? (transforms.shape[0] - info.start_instance)
              : (int)draw_instance_count;
    }
    info.mesh_attribute_info.mesh_attribute = transforms;
    info.mesh_attribute_info.has_attribute = transforms.valid;

    scene->mesh(info);
  }

  void point_light(py::tuple pos_, py::tuple color_) {
    glm::vec3 pos = tuple_to_vec3(pos_);
    glm::vec3 color = tuple_to_vec3(color_);
    scene->point_light(pos, color);
  }

  void ambient_light(py::tuple color_) {
    glm::vec3 color = tuple_to_vec3(color_);
    scene->ambient_light(color);
  }

  ~PyScene() {
    delete scene;
  }
};

struct PyCanvas {
  CanvasBase *canvas;  // not owned

  void set_background_color(py::tuple color_) {
    glm::vec3 color = tuple_to_vec3(color_);
    return canvas->set_background_color(color);
  }

  void set_image(FieldInfo img) {
    canvas->set_image({img});
  }

  void scene(PyScene &scene) {
    canvas->scene(scene.scene);
  }

  void triangles(FieldInfo vbo,
                 FieldInfo indices,
                 bool has_per_vertex_color,
                 py::tuple color_) {
    RenderableInfo renderable_info;
    renderable_info.vbo = vbo;
    renderable_info.indices = indices;
    renderable_info.has_per_vertex_color = has_per_vertex_color;

    TrianglesInfo info;
    info.renderable_info = renderable_info;
    info.color = tuple_to_vec3(color_);

    return canvas->triangles(info);
  }

  void lines(FieldInfo vbo,
             FieldInfo indices,
             bool has_per_vertex_color,
             py::tuple color_,
             float width) {
    RenderableInfo renderable_info;
    renderable_info.vbo = vbo;
    renderable_info.indices = indices;
    renderable_info.has_per_vertex_color = has_per_vertex_color;

    LinesInfo info;
    info.renderable_info = renderable_info;
    info.color = tuple_to_vec3(color_);
    info.width = width;

    return canvas->lines(info);
  }

  void circles(FieldInfo vbo,
               bool has_per_vertex_color,
               py::tuple color_,
               float radius) {
    RenderableInfo renderable_info;
    renderable_info.vbo = vbo;
    renderable_info.has_per_vertex_color = has_per_vertex_color;

    CirclesInfo info;
    info.renderable_info = renderable_info;
    info.color = tuple_to_vec3(color_);
    info.radius = radius;

    return canvas->circles(info);
  }
};

struct PyWindow {
  std::unique_ptr<WindowBase> window{nullptr};

  PyWindow(Program *prog,
           std::string name,
           py::tuple res,
           bool vsync,
           bool show_window,
           std::string package_path,
           Arch ti_arch,
           bool is_packed_mode) {
    AppConfig config = {name,    res[0].cast<int>(), res[1].cast<int>(),
                        vsync,   show_window,        package_path,
                        ti_arch, is_packed_mode};
    // todo: support other ggui backends
    if (!(taichi::arch_is_cpu(ti_arch) || ti_arch == Arch::vulkan ||
          ti_arch == Arch::cuda)) {
      throw std::runtime_error(
          "GGUI is only supported on cpu, vulkan and cuda backends");
    }
    if (!lang::vulkan::is_vulkan_api_available()) {
      throw std::runtime_error("Vulkan must be available for GGUI");
    }
    window = std::make_unique<vulkan::Window>(prog, config);
  }

  py::tuple get_window_shape() {
    auto [w, h] = window->get_window_shape();
    return pybind11::make_tuple(w, h);
  }

  void write_image(const std::string &filename) {
    window->write_image(filename);
  }

  void copy_depth_buffer_to_ndarray(Ndarray *depth_arr) {
    window->copy_depth_buffer_to_ndarray(*depth_arr);
  }

  py::array_t<float> get_image_buffer() {
    uint32_t w, h;
    auto &img_buffer = window->get_image_buffer(w, h);

    float *image = new float[w * h * 4];
    // Here we must match the numpy 3d array memory layout. Refs:
    // https://numpy.org/doc/stable/reference/arrays.ndarray.html
    for (int i = 0; i < w; i++) {
      for (int j = 0; j < h; j++) {
        auto pixel = img_buffer[j * w + i];
        for (int k = 0; k < 4; k++) {
          // must flip up-down to match the numpy array memory layout
          image[i * h * 4 + (h - j - 1) * 4 + k] = (pixel & 0xFF) / 255.0;
          pixel >>= 8;
        }
      }
    }
    // Here we must pass a deconstructor to free the memory in python scope.
    // Refs:
    // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
    py::capsule free_imgae(image, [](void *tmp) {
      float *image = reinterpret_cast<float *>(tmp);
      delete[] image;
    });

    return py::array_t<float>(
        py::detail::any_container<ssize_t>({w, h, 4}),
        py::detail::any_container<ssize_t>(
            {sizeof(float) * h * 4, sizeof(float) * 4, sizeof(float)}),
        image, free_imgae);
  }

  void show() {
    window->show();
  }

  bool is_pressed(std::string button) {
    return window->is_pressed(button);
  }

  bool is_running() {
    return window->is_running();
  }

  void set_is_running(bool value) {
    return window->set_is_running(value);
  }

  py::list get_events(EventType tag) {
    return py::cast(window->get_events(tag));
  }

  bool get_event(EventType e) {
    return window->get_event(e);
  }

  Event get_current_event() {
    return window->get_current_event();
  }
  void set_current_event(const Event &event) {
    window->set_current_event(event);
  }

  PyCanvas get_canvas() {
    PyCanvas canvas = {window->get_canvas()};
    return canvas;
  }

  PyGui GUI() {
    PyGui gui = {window->GUI()};
    return gui;
  }

  // this is so that the GUI class does not need to use any pybind related stuff
  py::tuple py_get_cursor_pos() {
    auto pos = window->get_cursor_pos();
    float x = std::get<0>(pos);
    float y = std::get<1>(pos);
    return py::make_tuple(x, y);
  }

  void destroy() {
    if (window) {
      window.reset();
    }
  }
};

void export_ggui(py::module &m) {
  m.attr("GGUI_AVAILABLE") = py::bool_(true);

  py::class_<PyWindow>(m, "PyWindow")
      .def(py::init<Program *, std::string, py::tuple, bool, bool, std::string,
                    Arch, bool>())
      .def("get_canvas", &PyWindow::get_canvas)
      .def("show", &PyWindow::show)
      .def("get_window_shape", &PyWindow::get_window_shape)
      .def("write_image", &PyWindow::write_image)
      .def("copy_depth_buffer_to_ndarray",
           &PyWindow::copy_depth_buffer_to_ndarray)
      .def("get_image_buffer_as_numpy", &PyWindow::get_image_buffer)
      .def("is_pressed", &PyWindow::is_pressed)
      .def("get_cursor_pos", &PyWindow::py_get_cursor_pos)
      .def("is_running", &PyWindow::is_running)
      .def("set_is_running", &PyWindow::set_is_running)
      .def("get_event", &PyWindow::get_event)
      .def("get_events", &PyWindow::get_events)
      .def("get_current_event", &PyWindow::get_current_event)
      .def("set_current_event", &PyWindow::set_current_event)
      .def("destroy", &PyWindow::destroy)
      .def("GUI", &PyWindow::GUI);

  py::class_<PyCanvas>(m, "PyCanvas")
      .def("set_background_color", &PyCanvas::set_background_color)
      .def("set_image", &PyCanvas::set_image)
      .def("triangles", &PyCanvas::triangles)
      .def("lines", &PyCanvas::lines)
      .def("circles", &PyCanvas::circles)
      .def("scene", &PyCanvas::scene);

  py::class_<PyGui>(m, "PyGui")
      .def("begin", &PyGui::begin)
      .def("end", &PyGui::end)
      .def("text", &PyGui::text)
      .def("checkbox", &PyGui::checkbox)
      .def("slider_int", &PyGui::slider_int)
      .def("slider_float", &PyGui::slider_float)
      .def("color_edit_3", &PyGui::color_edit_3)
      .def("button", &PyGui::button);

  py::class_<PyScene>(m, "PyScene")
      .def(py::init<>())
      .def("set_camera", &PyScene::set_camera)
      .def("lines", &PyScene::lines)
      .def("mesh", &PyScene::mesh)
      .def("particles", &PyScene::particles)
      .def("mesh_instance", &PyScene::mesh_instance)
      .def("point_light", &PyScene::point_light)
      .def("ambient_light", &PyScene::ambient_light);

  py::class_<PyCamera>(m, "PyCamera")
      .def(py::init<>())
      .def("lookat", &PyCamera::lookat)
      .def("position", &PyCamera::position)
      .def("up", &PyCamera::up)
      .def("projection_mode", &PyCamera::projection_mode)
      .def("fov", &PyCamera::fov)
      .def("left", &PyCamera::left)
      .def("right", &PyCamera::right)
      .def("top", &PyCamera::top)
      .def("bottom", &PyCamera::bottom)
      .def("z_near", &PyCamera::z_near)
      .def("z_far", &PyCamera::z_far)
      .def("get_view_matrix", &PyCamera::get_view_matrix)
      .def("get_projection_matrix", &PyCamera::get_projection_matrix);

  py::class_<Event>(m, "Event")
      .def_property("key", &Event::get_key, &Event::set_key);

  py::class_<FieldInfo>(m, "FieldInfo")
      .def(py::init<>())
      .def_property("field_type", &FieldInfo::get_field_type,
                    &FieldInfo::set_field_type)
      .def_property("matrix_rows", &FieldInfo::get_matrix_rows,
                    &FieldInfo::set_matrix_rows)
      .def_property("matrix_cols", &FieldInfo::get_field_type,
                    &FieldInfo::set_matrix_cols)
      .def_property("dtype", &FieldInfo::get_dtype, &FieldInfo::set_dtype)
      .def_property("field_source", &FieldInfo::get_field_source,
                    &FieldInfo::set_field_source)
      .def_property("snode", &FieldInfo::get_snode, &FieldInfo::set_snode)
      .def_property("shape", &FieldInfo::get_shape, &FieldInfo::set_shape)
      .def_property("valid", &FieldInfo::get_valid, &FieldInfo::set_valid);

  py::enum_<EventType>(m, "EventType")
      .value("Any", EventType::Any)
      .value("Press", EventType::Press)
      .value("Release", EventType::Release)
      .export_values();

  py::enum_<FieldSource>(m, "FieldSource")
      .value("TaichiCuda", FieldSource::TaichiCuda)
      .value("TaichiX64", FieldSource::TaichiX64)
      .value("TaichiVulkan", FieldSource::TaichiVulkan)
      .export_values();

  py::enum_<FieldType>(m, "FieldType")
      .value("Scalar", FieldType::Scalar)
      .value("Matrix", FieldType::Matrix)
      .export_values();

  py::enum_<ProjectionMode>(m, "ProjectionMode")
      .value("Perspective", ProjectionMode::Perspective)
      .value("Orthogonal", ProjectionMode::Orthogonal)
      .export_values();

  py::enum_<taichi::lang::PolygonMode>(m, "DisplayMode")
      .value("Fill", taichi::lang::PolygonMode::Fill)
      .value("Line", taichi::lang::PolygonMode::Line)
      .value("Point", taichi::lang::PolygonMode::Point)
      .export_values();
}

TI_UI_NAMESPACE_END

TI_NAMESPACE_BEGIN

void export_ggui(py::module &m) {
  ui::export_ggui(m);
}

TI_NAMESPACE_END

#else

TI_NAMESPACE_BEGIN

void export_ggui(py::module &m) {
  m.attr("GGUI_AVAILABLE") = py::bool_(false);
}

TI_NAMESPACE_END

#endif
