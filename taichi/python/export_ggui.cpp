
#include <unordered_map>
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
#include "taichi/ui/ggui/window.h"
#include "taichi/ui/common/canvas_base.h"
#include "taichi/ui/common/camera.h"
#include "taichi/ui/ggui/canvas.h"
#include "taichi/ui/ggui/scene.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/arch.h"
#include "taichi/program/field_info.h"
#include "taichi/ui/common/gui_base.h"
#include "taichi/program/ndarray.h"
#include <memory>

#ifndef TI_WITH_LLVM
#if defined(_WIN64)
typedef signed __int64 ssize_t;
#endif /* _WIN64 */
#endif /* TI_WITH_LLVM */

namespace taichi::ui {

using namespace taichi::lang;

glm::vec2 tuple_to_vec2(pybind11::tuple t) {
  return glm::vec2(t[0].cast<float>(), t[1].cast<float>());
}

glm::vec3 tuple_to_vec3(pybind11::tuple t) {
  return glm::vec3(t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>());
}

glm::vec4 tuple_to_vec4(pybind11::tuple t) {
  return glm::vec4(t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>(),
                   t[3].cast<float>());
}

glm::ivec2 tuple_to_ivec2(pybind11::tuple t) {
  return glm::ivec2(t[0].cast<int>(), t[1].cast<int>());
}

glm::ivec3 tuple_to_ivec3(pybind11::tuple t) {
  return glm::ivec3(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>());
}

glm::ivec4 tuple_to_ivec4(pybind11::tuple t) {
  return glm::ivec4(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>(),
                    t[3].cast<int>());
}

pybind11::tuple vec2_to_tuple(glm::vec2 v) {
  return pybind11::make_tuple(v.x, v.y);
}

pybind11::tuple vec3_to_tuple(glm::vec3 v) {
  return pybind11::make_tuple(v.x, v.y, v.z);
}

pybind11::tuple vec4_to_tuple(glm::vec4 v) {
  return pybind11::make_tuple(v.x, v.y, v.z, v.w);
}

pybind11::tuple ivec2_to_tuple(glm::ivec2 v) {
  return pybind11::make_tuple(v.x, v.y);
}

pybind11::tuple ivec3_to_tuple(glm::ivec3 v) {
  return pybind11::make_tuple(v.x, v.y, v.z);
}

pybind11::tuple ivec4_to_tuple(glm::ivec4 v) {
  return pybind11::make_tuple(v.x, v.y, v.z, v.w);
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
  GuiBase *gui = nullptr;  // not owned

  // Cache for string list items (combo, listbox): label -> cached data
  // Frame-based cleanup removes entries not used since last frame
  struct StringListCache {
    py::tuple items_tuple;                 // for identity comparison
    std::vector<std::string> items_str;    // owns the string data
    std::vector<const char *> items_cstr;  // points into items_str
    bool touched = false;                  // used this frame?
  };
  std::unordered_map<std::string, StringListCache> string_list_cache_;

  // Get cached C string pointers for a tuple of Python strings.
  // Rebuilds cache if tuple identity changed; marks entry as touched.
  const std::vector<const char *> &get_cached_strings_(const std::string &label,
                                                       py::tuple items_py) {
    auto it = string_list_cache_.find(label);
    if (it == string_list_cache_.end() ||
        !it->second.items_tuple.is(items_py)) {
      StringListCache cache;
      cache.items_tuple = items_py;
      for (auto item : items_py) {
        cache.items_str.push_back(item.cast<std::string>());
      }
      for (const auto &s : cache.items_str) {
        cache.items_cstr.push_back(s.c_str());
      }
      string_list_cache_[label] = std::move(cache);
      it = string_list_cache_.find(label);
    }
    it->second.touched = true;
    return it->second.items_cstr;
  }

  void begin(std::string name, float x, float y, float width, float height) {
    gui->begin(name, x, y, width, height);
  }
  void end() {
    gui->end();
  }
  void text(std::string text) {
    gui->text(text);
  }
  void text_colored(std::string text, py::tuple color) {
    gui->text(text, tuple_to_vec3(color));
  }
  bool checkbox(std::string name, bool old_value) {
    return gui->checkbox(name, old_value);
  }
  int slider_int(std::string name, int old_value, int minimum, int maximum) {
    return gui->slider_int(name, old_value, minimum, maximum);
  }
  py::tuple slider_int2(std::string name,
                        py::tuple old_value,
                        int minimum,
                        int maximum) {
    return ivec2_to_tuple(
        gui->slider_int2(name, tuple_to_ivec2(old_value), minimum, maximum));
  }
  py::tuple slider_int3(std::string name,
                        py::tuple old_value,
                        int minimum,
                        int maximum) {
    return ivec3_to_tuple(
        gui->slider_int3(name, tuple_to_ivec3(old_value), minimum, maximum));
  }
  py::tuple slider_int4(std::string name,
                        py::tuple old_value,
                        int minimum,
                        int maximum) {
    return ivec4_to_tuple(
        gui->slider_int4(name, tuple_to_ivec4(old_value), minimum, maximum));
  }
  float slider_float(std::string name,
                     float old_value,
                     float minimum,
                     float maximum) {
    return gui->slider_float(name, old_value, minimum, maximum);
  }
  py::tuple slider_float2(std::string name,
                          py::tuple old_value,
                          float minimum,
                          float maximum) {
    return vec2_to_tuple(
        gui->slider_float2(name, tuple_to_vec2(old_value), minimum, maximum));
  }
  py::tuple slider_float3(std::string name,
                          py::tuple old_value,
                          float minimum,
                          float maximum) {
    return vec3_to_tuple(
        gui->slider_float3(name, tuple_to_vec3(old_value), minimum, maximum));
  }
  py::tuple slider_float4(std::string name,
                          py::tuple old_value,
                          float minimum,
                          float maximum) {
    return vec4_to_tuple(
        gui->slider_float4(name, tuple_to_vec4(old_value), minimum, maximum));
  }
  py::tuple color_edit_3(std::string name, py::tuple old_value) {
    glm::vec3 old_color = tuple_to_vec3(old_value);
    glm::vec3 new_color = gui->color_edit_3(name, old_color);
    return vec3_to_tuple(new_color);
  }
  py::tuple color_edit_4(std::string name, py::tuple old_value) {
    glm::vec4 old_color = tuple_to_vec4(old_value);
    glm::vec4 new_color = gui->color_edit_4(name, old_color);
    return vec4_to_tuple(new_color);
  }
  py::tuple color_picker_3(std::string name, py::tuple old_value) {
    glm::vec3 old_color = tuple_to_vec3(old_value);
    glm::vec3 new_color = gui->color_picker_3(name, old_color);
    return vec3_to_tuple(new_color);
  }
  py::tuple color_picker_4(std::string name, py::tuple old_value) {
    glm::vec4 old_color = tuple_to_vec4(old_value);
    glm::vec4 new_color = gui->color_picker_4(name, old_color);
    return vec4_to_tuple(new_color);
  }
  bool button(std::string name) {
    return gui->button(name);
  }
  int input_int(std::string label, int old_value) {
    return gui->input_int(label, old_value);
  }
  py::tuple input_int2(std::string label, py::tuple old_value) {
    return ivec2_to_tuple(gui->input_int2(label, tuple_to_ivec2(old_value)));
  }
  py::tuple input_int3(std::string label, py::tuple old_value) {
    return ivec3_to_tuple(gui->input_int3(label, tuple_to_ivec3(old_value)));
  }
  py::tuple input_int4(std::string label, py::tuple old_value) {
    return ivec4_to_tuple(gui->input_int4(label, tuple_to_ivec4(old_value)));
  }
  float input_float(std::string label, float old_value) {
    return gui->input_float(label, old_value);
  }
  py::tuple input_float2(std::string label, py::tuple old_value) {
    return vec2_to_tuple(gui->input_float2(label, tuple_to_vec2(old_value)));
  }
  py::tuple input_float3(std::string label, py::tuple old_value) {
    return vec3_to_tuple(gui->input_float3(label, tuple_to_vec3(old_value)));
  }
  py::tuple input_float4(std::string label, py::tuple old_value) {
    return vec4_to_tuple(gui->input_float4(label, tuple_to_vec4(old_value)));
  }
  int drag_int(std::string label,
               int old_value,
               float speed,
               int minimum,
               int maximum) {
    return gui->drag_int(label, old_value, speed, minimum, maximum);
  }
  py::tuple drag_int2(std::string label,
                      py::tuple old_value,
                      float speed,
                      int minimum,
                      int maximum) {
    return ivec2_to_tuple(gui->drag_int2(label, tuple_to_ivec2(old_value),
                                         speed, minimum, maximum));
  }
  py::tuple drag_int3(std::string label,
                      py::tuple old_value,
                      float speed,
                      int minimum,
                      int maximum) {
    return ivec3_to_tuple(gui->drag_int3(label, tuple_to_ivec3(old_value),
                                         speed, minimum, maximum));
  }
  py::tuple drag_int4(std::string label,
                      py::tuple old_value,
                      float speed,
                      int minimum,
                      int maximum) {
    return ivec4_to_tuple(gui->drag_int4(label, tuple_to_ivec4(old_value),
                                         speed, minimum, maximum));
  }
  float drag_float(std::string label,
                   float old_value,
                   float speed,
                   float minimum,
                   float maximum) {
    return gui->drag_float(label, old_value, speed, minimum, maximum);
  }
  py::tuple drag_float2(std::string label,
                        py::tuple old_value,
                        float speed,
                        float minimum,
                        float maximum) {
    return vec2_to_tuple(gui->drag_float2(label, tuple_to_vec2(old_value),
                                          speed, minimum, maximum));
  }
  py::tuple drag_float3(std::string label,
                        py::tuple old_value,
                        float speed,
                        float minimum,
                        float maximum) {
    return vec3_to_tuple(gui->drag_float3(label, tuple_to_vec3(old_value),
                                          speed, minimum, maximum));
  }
  py::tuple drag_float4(std::string label,
                        py::tuple old_value,
                        float speed,
                        float minimum,
                        float maximum) {
    return vec4_to_tuple(gui->drag_float4(label, tuple_to_vec4(old_value),
                                          speed, minimum, maximum));
  }
  bool tree_node_push(std::string label) {
    return gui->tree_node_push(label);
  }
  void tree_node_pop() {
    gui->tree_node_pop();
  }
  void separator() {
    gui->separator();
  }
  void same_line() {
    gui->same_line();
  }
  void indent() {
    gui->indent();
  }
  void unindent() {
    gui->unindent();
  }
  void progress_bar(float fraction) {
    gui->progress_bar(fraction);
  }
  bool collapsing_header(std::string label) {
    return gui->collapsing_header(label);
  }
  bool selectable(std::string label, bool selected) {
    return gui->selectable(label, selected);
  }
  bool radio_button(std::string label, bool active) {
    return gui->radio_button(label, active);
  }
  bool begin_tab_bar(std::string id) {
    return gui->begin_tab_bar(id);
  }
  void end_tab_bar() {
    gui->end_tab_bar();
  }
  bool begin_tab_item(std::string label) {
    return gui->begin_tab_item(label);
  }
  void end_tab_item() {
    gui->end_tab_item();
  }
  bool begin_table(std::string id, int columns) {
    return gui->begin_table(id, columns);
  }
  void end_table() {
    gui->end_table();
  }
  void table_setup_column(std::string label) {
    gui->table_setup_column(label);
  }
  void table_headers_row() {
    gui->table_headers_row();
  }
  void table_next_row() {
    gui->table_next_row();
  }
  bool table_next_column() {
    return gui->table_next_column();
  }
  int combo(std::string label, int current_item, py::tuple items_py) {
    const auto &items = get_cached_strings_(label, items_py);
    return gui->combo(label, current_item, items);
  }

  int listbox(std::string label,
              int current_item,
              py::tuple items_py,
              int height_in_items) {
    const auto &items = get_cached_strings_(label, items_py);
    return gui->listbox(label, current_item, items, height_in_items);
  }

  // Called at frame end to clean up stale cache entries
  void frame_end() {
    for (auto it = string_list_cache_.begin();
         it != string_list_cache_.end();) {
      if (!it->second.touched) {
        it = string_list_cache_.erase(it);
      } else {
        it->second.touched = false;
        ++it;
      }
    }
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
                 bool has_per_vertex_radius,
                 py::tuple color_,
                 float radius,
                 float draw_vertex_count,
                 float draw_first_vertex) {
    RenderableInfo renderable_info;
    renderable_info.vbo = vbo;
    renderable_info.has_user_customized_draw = true;
    renderable_info.has_per_vertex_color = has_per_vertex_color;
    renderable_info.has_per_vertex_radius = has_per_vertex_radius;
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

struct PySceneV2 {
  SceneBase *scene;  // not owned

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
                 bool has_per_vertex_radius,
                 py::tuple color_,
                 float radius,
                 float draw_vertex_count,
                 float draw_first_vertex) {
    RenderableInfo renderable_info;
    renderable_info.vbo = vbo;
    renderable_info.has_user_customized_draw = true;
    renderable_info.has_per_vertex_color = has_per_vertex_color;
    renderable_info.has_per_vertex_radius = has_per_vertex_radius;
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

  void set_image_texture(Texture *texture) {
    canvas->set_image(texture);
  }

  void scene(PyScene &scene) {
    canvas->scene(scene.scene);
  }

  void scene_v2(PySceneV2 &scene) {
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
               bool has_per_vertex_radius,
               py::tuple color_,
               float radius) {
    RenderableInfo renderable_info;
    renderable_info.vbo = vbo;
    renderable_info.has_per_vertex_color = has_per_vertex_color;
    renderable_info.has_per_vertex_radius = has_per_vertex_radius;

    CirclesInfo info;
    info.renderable_info = renderable_info;
    info.color = tuple_to_vec3(color_);
    info.radius = radius;

    return canvas->circles(info);
  }
};

struct PyWindow {
  std::unique_ptr<WindowBase> window{nullptr};
  std::unique_ptr<PyGui> py_gui_{nullptr};

  PyWindow(Program *prog,
           std::string name,
           py::tuple res,
           py::tuple pos,
           bool vsync,
           bool show_window,
           double fps_limit,
           std::string package_path,
           Arch ti_arch) {
    Arch ggui_arch = Arch::vulkan;

    if (ti_arch == Arch::metal) {
      ggui_arch = Arch::metal;
    }

    if (ggui_arch == Arch::vulkan) {
      // Verify vulkan available
      if (!lang::vulkan::is_vulkan_api_available()) {
        throw std::runtime_error("Vulkan must be available for GGUI");
      }
    }

    AppConfig config = {name,
                        res[0].cast<int>(),
                        res[1].cast<int>(),
                        pos[0].cast<int>(),
                        pos[1].cast<int>(),
                        vsync,
                        show_window,
                        fps_limit,
                        package_path,
                        ti_arch,
                        ggui_arch};

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
    if (py_gui_) {
      py_gui_->frame_end();
    }
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

  PySceneV2 get_scene() {
    PySceneV2 scene = {window->get_scene()};
    return scene;
  }

  PyGui &gui() {
    if (!py_gui_) {
      py_gui_ = std::make_unique<PyGui>();
      py_gui_->gui = window->gui();
    }
    return *py_gui_;
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
      .def(py::init<Program *, std::string, py::tuple, py::tuple, bool, bool,
                    double, std::string, Arch>())
      .def("get_canvas", &PyWindow::get_canvas)
      .def("get_scene", &PyWindow::get_scene)
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
      .def("GUI", &PyWindow::gui);

  py::class_<PyCanvas>(m, "PyCanvas")
      .def("set_background_color", &PyCanvas::set_background_color)
      .def("set_image", &PyCanvas::set_image)
      .def("set_image_texture", &PyCanvas::set_image_texture)
      .def("triangles", &PyCanvas::triangles)
      .def("lines", &PyCanvas::lines)
      .def("circles", &PyCanvas::circles)
      .def("scene", &PyCanvas::scene)
      .def("scene_v2", &PyCanvas::scene_v2);

  py::class_<PyGui>(m, "PyGui")
      .def("begin", &PyGui::begin)
      .def("end", &PyGui::end)
      .def("text", &PyGui::text)
      .def("text_colored", &PyGui::text_colored)
      .def("checkbox", &PyGui::checkbox)
      .def("slider_int", &PyGui::slider_int)
      .def("slider_int2", &PyGui::slider_int2)
      .def("slider_int3", &PyGui::slider_int3)
      .def("slider_int4", &PyGui::slider_int4)
      .def("slider_float", &PyGui::slider_float)
      .def("slider_float2", &PyGui::slider_float2)
      .def("slider_float3", &PyGui::slider_float3)
      .def("slider_float4", &PyGui::slider_float4)
      .def("color_edit_3", &PyGui::color_edit_3)
      .def("color_edit_4", &PyGui::color_edit_4)
      .def("color_picker_3", &PyGui::color_picker_3)
      .def("color_picker_4", &PyGui::color_picker_4)
      .def("button", &PyGui::button)
      .def("input_int", &PyGui::input_int)
      .def("input_int2", &PyGui::input_int2)
      .def("input_int3", &PyGui::input_int3)
      .def("input_int4", &PyGui::input_int4)
      .def("input_float", &PyGui::input_float)
      .def("input_float2", &PyGui::input_float2)
      .def("input_float3", &PyGui::input_float3)
      .def("input_float4", &PyGui::input_float4)
      .def("drag_int", &PyGui::drag_int)
      .def("drag_int2", &PyGui::drag_int2)
      .def("drag_int3", &PyGui::drag_int3)
      .def("drag_int4", &PyGui::drag_int4)
      .def("drag_float", &PyGui::drag_float)
      .def("drag_float2", &PyGui::drag_float2)
      .def("drag_float3", &PyGui::drag_float3)
      .def("drag_float4", &PyGui::drag_float4)
      .def("tree_node_push", &PyGui::tree_node_push)
      .def("tree_node_pop", &PyGui::tree_node_pop)
      .def("separator", &PyGui::separator)
      .def("same_line", &PyGui::same_line)
      .def("indent", &PyGui::indent)
      .def("unindent", &PyGui::unindent)
      .def("progress_bar", &PyGui::progress_bar)
      .def("combo", &PyGui::combo)
      .def("collapsing_header", &PyGui::collapsing_header)
      .def("selectable", &PyGui::selectable)
      .def("radio_button", &PyGui::radio_button)
      .def("listbox", &PyGui::listbox)
      .def("begin_tab_bar", &PyGui::begin_tab_bar)
      .def("end_tab_bar", &PyGui::end_tab_bar)
      .def("begin_tab_item", &PyGui::begin_tab_item)
      .def("end_tab_item", &PyGui::end_tab_item)
      .def("begin_table", &PyGui::begin_table)
      .def("end_table", &PyGui::end_table)
      .def("table_setup_column", &PyGui::table_setup_column)
      .def("table_headers_row", &PyGui::table_headers_row)
      .def("table_next_row", &PyGui::table_next_row)
      .def("table_next_column", &PyGui::table_next_column);

  py::class_<PyScene>(m, "PyScene")
      .def(py::init<>())
      .def("set_camera", &PyScene::set_camera)
      .def("lines", &PyScene::lines)
      .def("mesh", &PyScene::mesh)
      .def("particles", &PyScene::particles)
      .def("mesh_instance", &PyScene::mesh_instance)
      .def("point_light", &PyScene::point_light)
      .def("ambient_light", &PyScene::ambient_light);

  py::class_<PySceneV2>(m, "PySceneV2")
      .def("set_camera", &PySceneV2::set_camera)
      .def("lines", &PySceneV2::lines)
      .def("mesh", &PySceneV2::mesh)
      .def("particles", &PySceneV2::particles)
      .def("mesh_instance", &PySceneV2::mesh_instance)
      .def("point_light", &PySceneV2::point_light)
      .def("ambient_light", &PySceneV2::ambient_light);

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
      .def_property("valid", &FieldInfo::get_valid, &FieldInfo::set_valid)
      .def_property("num_elements", &FieldInfo::get_num_elements,
                    &FieldInfo::set_num_elements)
      .def_property("shape", &FieldInfo::get_shape, &FieldInfo::set_shape)
      .def_property("field_source", &FieldInfo::get_field_source,
                    &FieldInfo::set_field_source)
      .def_property("dtype", &FieldInfo::get_dtype, &FieldInfo::set_dtype)
      .def_property("dev_alloc", &FieldInfo::get_dev_alloc,
                    &FieldInfo::set_dev_alloc);

  py::enum_<EventType>(m, "EventType")
      .value("Any", EventType::Any)
      .value("Press", EventType::Press)
      .value("Release", EventType::Release)
      .export_values();

  py::enum_<FieldSource>(m, "FieldSource")
      .value("TaichiNDarray", FieldSource::TaichiNDarray)
      .value("HostMappedPtr", FieldSource::HostMappedPtr)
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

}  // namespace taichi::ui

namespace taichi {

void export_ggui(py::module &m) {
  ui::export_ggui(m);
}

}  // namespace taichi

#else

namespace taichi {

void export_ggui(py::module &m) {
  m.attr("GGUI_AVAILABLE") = py::bool_(false);
}

}  // namespace taichi

#endif
