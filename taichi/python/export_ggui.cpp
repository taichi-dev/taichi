
#include <vector>
#include "pybind11/pybind11.h"
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
#include "taichi/ui/common/field_info.h"
#include "taichi/ui/common/gui_base.h"
#include <memory>

TI_UI_NAMESPACE_BEGIN

using namespace taichi::lang;

glm::vec3 tuple_to_vec3(pybind11::tuple t) {
  return glm::vec3(t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>());
}

pybind11::tuple vec3_to_tuple(glm::vec3 v) {
  return pybind11::make_tuple(v.x, v.y, v.z);
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

  void mesh(FieldInfo vbo,
            bool has_per_vertex_color,
            FieldInfo indices,
            py::tuple color,
            bool two_sided) {
    RenderableInfo renderable_info;
    renderable_info.vbo = vbo;
    renderable_info.has_per_vertex_color = has_per_vertex_color;
    renderable_info.indices = indices;

    MeshInfo info;
    info.renderable_info = renderable_info;
    info.color = tuple_to_vec3(color);
    info.two_sided = two_sided;

    scene->mesh(info);
  }

  void particles(FieldInfo vbo,
                 bool has_per_vertex_color,
                 py::tuple color_,
                 float radius) {
    RenderableInfo renderable_info;
    renderable_info.vbo = vbo;
    renderable_info.has_per_vertex_color = has_per_vertex_color;

    ParticlesInfo info;
    info.renderable_info = renderable_info;
    info.color = tuple_to_vec3(color_);
    info.radius = radius;

    scene->particles(info);
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
    window = std::make_unique<vulkan::Window>(prog, config);
  }

  void write_image(const std::string &filename) {
    window->write_image(filename);
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
      .def(py::init<Program*, std::string, py::tuple, bool, bool, std::string, Arch,
                    bool>())
      .def("get_canvas", &PyWindow::get_canvas)
      .def("show", &PyWindow::show)
      .def("write_image", &PyWindow::write_image)
      .def("is_pressed", &PyWindow::is_pressed)
      .def("get_cursor_pos", &PyWindow::py_get_cursor_pos)
      .def("is_running", &PyWindow::is_running)
      .def("set_is_running", &PyWindow::set_is_running)
      .def("get_event", &PyWindow::get_event)
      .def("get_events", &PyWindow::get_events)
      .def_property("event", &PyWindow::get_current_event,
                    &PyWindow::set_current_event)
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
      .def("slider_float", &PyGui::slider_float)
      .def("color_edit_3", &PyGui::color_edit_3)
      .def("button", &PyGui::button);

  py::class_<PyScene>(m, "PyScene")
      .def(py::init<>())
      .def("set_camera", &PyScene::set_camera)
      .def("mesh", &PyScene::mesh)
      .def("particles", &PyScene::particles)
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
      .def("z_far", &PyCamera::z_far);

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
