/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/python/export.h"
#include "taichi/util/image_io.h"
#include "taichi/gui/gui.h"

TI_NAMESPACE_BEGIN

void export_visual(py::module &m) {
  // GUI
  using Line = Canvas::Line;
  using Circle = Canvas::Circle;
  py::class_<GUI>(m, "GUI")
      .def(py::init<std::string, Vector2i>())
      .def("get_canvas", &GUI::get_canvas, py::return_value_policy::reference)
      .def("set_img",
           [&](GUI *gui, std::size_t ptr) {
             auto &img = gui->canvas->img;
             std::memcpy((void *)img.get_data().data(), (void *)ptr,
                         img.get_data_size());
           })
      .def("screenshot", &GUI::screenshot)
      .def("has_key_event", &GUI::has_key_event)
      .def("wait_key_event", &GUI::wait_key_event)
      .def("get_key_event_head_key", &GUI::get_key_event_head_key)
      .def("get_key_event_head_type", &GUI::get_key_event_head_type)
      .def("get_key_event_head_pos", &GUI::get_key_event_head_pos)
      .def("pop_key_event_head", &GUI::pop_key_event_head)
      .def("get_cursor_pos", &GUI::get_cursor_pos)
      .def_readwrite("title", &GUI::window_name)
      .def("set_profiler",
           [](GUI *gui, void *profiler) -> void {
             gui->set_profiler((lang::ProfilerBase *)profiler);
           })
      .def("update", &GUI::update);
  py::class_<Canvas>(m, "Canvas")
      .def("clear", static_cast<void (Canvas::*)(uint32)>(&Canvas::clear))
      .def("rect", &Canvas::rect, py::return_value_policy::reference)
      .def("path",
           static_cast<Line &(Canvas::*)(Vector2, Vector2)>(&Canvas::path),
           py::return_value_policy::reference)
      .def("path_single", &Canvas::path_single)
      .def("triangle_single", &Canvas::triangle_single)
      .def("circles_batched", &Canvas::circles_batched)
      .def("circle_single", &Canvas::circle_single)
      .def("circle", static_cast<Circle &(Canvas::*)(Vector2)>(&Canvas::circle),
           py::return_value_policy::reference);
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
