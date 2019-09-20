/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/python/export.h>

#include <taichi/math/sdf.h>

#include <taichi/visual/camera.h>
#include <taichi/visual/renderer.h>
#include <taichi/visual/volume_material.h>
#include <taichi/visual/surface_material.h>
#include <taichi/visual/envmap.h>
#include <taichi/visualization/particle_visualization.h>
#include <taichi/common/asset_manager.h>

#include <taichi/geometry/factory.h>
#include <taichi/math/levelset.h>
#include <taichi/visual/gui.h>

PYBIND11_MAKE_OPAQUE(std::vector<taichi::RenderParticle>);
PYBIND11_MAKE_OPAQUE(std::vector<taichi::Triangle>);

TC_NAMESPACE_BEGIN

template <typename T, typename V>
V address_as(T &obj) {
  return (V)(&obj);
};

template <int N, int M, typename T>
VectorFunction<N, M, T> function_from_py_obj(py::object func) {
  return [func](const VectorLengthed<N, T> &p) -> VectorLengthed<M, T> {
    // TODO: GIL here seems inefficient...
    PyGILState_STATE state = PyGILState_Ensure();
    py::function f = py::reinterpret_borrow<py::function>(func);
    VectorLengthed<M, T> ret = f(p).template cast<VectorLengthed<M, T>>();
    PyGILState_Release(state);
    return ret;
  };
}

std::vector<Triangle> merge_mesh(const std::vector<Triangle> &a,
                                 const std::vector<Triangle> &b) {
  std::vector<Triangle> merged = a;
  merged.insert(merged.end(), b.begin(), b.end());
  return merged;
}

void export_visual(py::module &m) {
  DEFINE_VECTOR_OF_NAMED(RenderParticle, "RenderParticles");
  DEFINE_VECTOR_OF_NAMED(Triangle, "Triangles");

  m.def("get_function11_address", address_as<Function11, uint64>);
  m.def("get_function12_address", address_as<Function12, uint64>);
  m.def("get_function13_address", address_as<Function13, uint64>);

  m.def("function11_from_py_obj", function_from_py_obj<1, 1, real>);
  m.def("function12_from_py_obj", function_from_py_obj<1, 2, real>);
  m.def("function13_from_py_obj", function_from_py_obj<1, 3, real>);

  m.def("function21_from_py_obj", function_from_py_obj<2, 1, real>);
  m.def("function22_from_py_obj", function_from_py_obj<2, 2, real>);
  m.def("function23_from_py_obj", function_from_py_obj<2, 3, real>);

  m.def("function31_from_py_obj", function_from_py_obj<3, 1, real>);
  m.def("function32_from_py_obj", function_from_py_obj<3, 2, real>);
  m.def("function33_from_py_obj", function_from_py_obj<3, 3, real>);

  // TODO: these should registered by iterating over existing interfaces.
  m.def("merge_mesh", merge_mesh);
  m.def("generate_mesh", Mesh3D::generate);
  m.def("rasterize_render_particles", rasterize_render_particles);
  m.def("create_mesh", std::make_shared<Mesh>);
  m.def("create_scene", std::make_shared<Scene>);

  py::class_<Texture, std::shared_ptr<Texture>>(m, "Texture")
      .def("initialize", &Texture::initialize)
      .def("rasterize",
           static_cast<Array2D<Vector4> (Texture::*)(int, int) const>(
               &Texture::rasterize))
      .def("rasterize3",
           static_cast<Array2D<Vector3> (Texture::*)(int, int) const>(
               &Texture::rasterize3));

  py::class_<VolumeMaterial, std::shared_ptr<VolumeMaterial>>(m,
                                                              "VolumeMaterial")
      .def("initialize", &VolumeMaterial::initialize);
  ;

  py::class_<SurfaceMaterial, std::shared_ptr<SurfaceMaterial>>(
      m, "SurfaceMaterial")
      .def("initialize", static_cast<void (SurfaceMaterial::*)(const Config &)>(
                             &SurfaceMaterial::initialize))
      .def("set_internal_material", &SurfaceMaterial::set_internal_material);

  py::class_<EnvironmentMap, std::shared_ptr<EnvironmentMap>>(m,
                                                              "EnvironmentMap")
      .def("initialize", &EnvironmentMap::initialize)
      .def("set_transform", &EnvironmentMap::set_transform);

  py::class_<Mesh, std::shared_ptr<Mesh>>(m, "Mesh")
      .def("initialize", &Mesh::initialize)
      .def("set_untransformed_triangles", &Mesh::set_untransformed_triangles)
      .def("set_material", &Mesh::set_material)
      .def_readwrite("transform", &Mesh::transform);

  py::class_<Scene, std::shared_ptr<Scene>>(m, "Scene")
      //.def("initialize", &Scene::initialize)
      .def("finalize", &Scene::finalize)
      .def("add_mesh", &Scene::add_mesh)
      .def("set_atmosphere_material", &Scene::set_atmosphere_material)
      .def("set_environment_map", &Scene::set_environment_map)
      .def("set_camera", &Scene::set_camera);

  // Renderers
  py::class_<Renderer, std::shared_ptr<Renderer>>(m, "Renderer")
      .def("initialize", &Renderer::initialize)
      .def("set_scene", &Renderer::set_scene)
      .def("render_stage", &Renderer::render_stage)
      .def("write_output", &Renderer::write_output)
      .def("get_output", &Renderer::get_output);

  py::class_<Camera, std::shared_ptr<Camera>>(m, "Camera")
      .def("initialize", &Camera::initialize);

  py::class_<ParticleRenderer, std::shared_ptr<ParticleRenderer>>(
      m, "ParticleRenderer")
      .def("initialize", &ParticleRenderer::initialize)
      .def("set_camera", &ParticleRenderer::set_camera)
      .def("render", &ParticleRenderer::render);

  py::class_<SDF, std::shared_ptr<SDF>>(m, "SDF")
      .def("initialize", &SDF::initialize)
      .def("eval", &SDF::eval);

  py::class_<Function11>(m, "Function11");
  py::class_<Function12>(m, "Function12");
  py::class_<Function13>(m, "Function13");
  py::class_<Function21>(m, "Function21");
  py::class_<Function22>(m, "Function22");
  py::class_<Function23>(m, "Function23");
  py::class_<Function31>(m, "Function31");
  py::class_<Function32>(m, "Function32");
  py::class_<Function33>(m, "Function33");

  // GUI
  using Line = Canvas::Line;
  using Circle = Canvas::Circle;
  py::class_<GUI>(m, "GUI")
      .def(py::init<std::string, Vector2i>())
      .def("get_canvas", &GUI::get_canvas, py::return_value_policy::reference)
      .def("update", &GUI::update);
  py::class_<Canvas>(m, "Canvas")
      .def("clear", static_cast<void (Canvas::*)(int)>(&Canvas::clear))
      .def("rect", &Canvas::rect, py::return_value_policy::reference)
      .def("circle", static_cast<Circle (Canvas::*)(Vector2)>(&Canvas::circle),
           py::return_value_policy::reference);
  py::class_<Line>(m, "Line")
      .def("radius", &Line::radius, py::return_value_policy::reference)
      .def("close", &Line::close, py::return_value_policy::reference)
      .def("color", static_cast<Line &(Line::*)(int)>(&Line::color),
           py::return_value_policy::reference);
  py::class_<Circle>(m, "Circle")
      .def("finish", &Circle::finish)
      .def("radius", &Circle::radius, py::return_value_policy::reference)
      .def("color", static_cast<Circle &(Circle::*)(int)>(&Circle::color),
           py::return_value_policy::reference);
}

TC_NAMESPACE_END
