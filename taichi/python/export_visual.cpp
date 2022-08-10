/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/python/export.h"
#include "taichi/util/image_io.h"
#include "taichi/ui/gui/gui.h"

TI_NAMESPACE_BEGIN

void export_visual(py::module &m) {
  // GUI
  using Line = Canvas::Line;
  using Circle = Canvas::Circle;
  using Type = GUI::KeyEvent::Type;

  auto key_event = py::class_<GUI::KeyEvent>(m, "KeyEvent");
  key_event.def_readonly("type", &GUI::KeyEvent::type)
      .def_readonly("key", &GUI::KeyEvent::key)
      .def_readonly("pos", &GUI::KeyEvent::pos)
      .def_readonly("delta", &GUI::KeyEvent::delta);
  py::enum_<GUI::KeyEvent::Type>(key_event, "EType")
      .value("Move", Type::move)
      .value("Press", Type::press)
      .value("Release", Type::release);
  py::class_<GUI>(m, "GUI")
      .def(py::init<std::string, Vector2i, bool, bool, bool, uintptr_t>())
      .def_readwrite("frame_delta_limit", &GUI::frame_delta_limit)
      .def_readwrite("should_close", &GUI::should_close)
      .def("get_canvas", &GUI::get_canvas, py::return_value_policy::reference)
      .def("set_img",
           [&](GUI *gui, std::size_t ptr) {
             auto &img = gui->canvas->img;
             std::memcpy((void *)img.get_data().data(), (void *)ptr,
                         img.get_data_size());
           })
      .def("get_img",
           [&](GUI *gui, std::size_t ptr) {
             auto &img = gui->canvas->img;
             std::memcpy((void *)ptr, (void *)img.get_data().data(),
                         img.get_data_size());
           })
      .def("screenshot", &GUI::screenshot)
      .def("set_widget_value",
           [](GUI *gui, int wid, float value) {
             *gui->widget_values.at(wid) = value;
           })
      .def("get_widget_value",
           [](GUI *gui, int wid) -> float {
             return *gui->widget_values.at(wid);
           })
      .def("make_slider",
           [](GUI *gui, std::string text, float init_value, float minimum,
              float maximum, float step) {
             auto val = std::make_unique<float>(init_value);
             auto val_ptr = val.get();
             gui->widget_values.push_back(std::move(val));
             gui->slider(text, *val_ptr, minimum, maximum, step);
             return gui->widget_values.size() - 1;
           })
      .def("make_label",
           [](GUI *gui, std::string text, float init_value) {
             auto val = std::make_unique<float>(init_value);
             auto val_ptr = val.get();
             gui->widget_values.push_back(std::move(val));
             gui->label(text, *val_ptr);
             return gui->widget_values.size() - 1;
           })
      .def("make_button",
           [](GUI *gui, std::string text, std::string event_name) {
             gui->button(text, [=]() {
               gui->key_events.push_back(GUI::KeyEvent{
                   GUI::KeyEvent::Type::press, event_name, gui->cursor_pos});
             });
           })
      .def("canvas_untransform", &GUI::canvas_untransform)
      .def("has_key_event", &GUI::has_key_event)
      .def("wait_key_event", &GUI::wait_key_event)
      .def("get_key_event_head", &GUI::get_key_event_head)
      .def("pop_key_event_head", &GUI::pop_key_event_head)
      .def("get_cursor_pos", &GUI::get_cursor_pos)
      .def_readwrite("title", &GUI::window_name)
      .def("update", &GUI::update);
  py::class_<Canvas>(m, "Canvas")
      .def("clear", static_cast<void (Canvas::*)(uint32)>(&Canvas::clear))
      .def("rect", &Canvas::rect, py::return_value_policy::reference)
      .def("path",
           static_cast<Line &(Canvas::*)(Vector2, Vector2)>(&Canvas::path),
           py::return_value_policy::reference)
      .def("path_single", &Canvas::path_single)
      .def("paths_batched", &Canvas::paths_batched)
      .def("triangle_single", &Canvas::triangle_single)
      .def("triangles_batched", &Canvas::triangles_batched)
      .def("circles_batched", &Canvas::circles_batched)
      .def("circle_single", &Canvas::circle_single)
      .def("circle", static_cast<Circle &(Canvas::*)(Vector2)>(&Canvas::circle),
           py::return_value_policy::reference)
      .def("text", &Canvas::text);
  py::class_<Line>(m, "Line")
      .def("finish", &Line::finish)
      .def("radius", &Line::radius, py::return_value_policy::reference)
      .def("close", &Line::close, py::return_value_policy::reference)
      .def("color", static_cast<Line &(Line::*)(int)>(&Line::color),
           py::return_value_policy::reference);
  py::class_<Circle>(m, "Circle")
      .def("finish", &Circle::finish)
      .def("radius", &Circle::radius, py::return_value_policy::reference)
      .def("color", static_cast<Circle &(Circle::*)(int)>(&Circle::color),
           py::return_value_policy::reference);
  m.def("imwrite", &imwrite);
  m.def("imread", &imread);
  // TODO(archibate): See misc/image.py
  m.def("C_memcpy", [](size_t dst, size_t src, size_t size) {
    std::memcpy((void *)dst, (void *)src, size);
  });
}

TI_NAMESPACE_END
