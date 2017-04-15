/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/python/export.h>

#include <taichi/visual/camera.h>
#include <taichi/visual/renderer.h>
#include <taichi/visual/volume_material.h>
#include <taichi/visual/surface_material.h>
#include <taichi/math/sdf.h>
#include <taichi/visualization/particle_visualization.h>
#include <taichi/common/asset_manager.h>

#include <taichi/geometry/factory.h>

PYBIND11_MAKE_OPAQUE(std::vector<taichi::RenderParticle>);
PYBIND11_MAKE_OPAQUE(std::vector<taichi::Triangle>);

TC_NAMESPACE_BEGIN

Function23 function23_from_py_obj(py::object func) {
    return [func](Vector2 p) -> Vector3 {
        // TODO: GIL here seems inefficient...
        PyGILState_STATE state = PyGILState_Ensure();
        py::function f = py::reinterpret_borrow<py::function>(func);
        Vector3 ret = f(p).cast<Vector3>();
        PyGILState_Release(state);
        return ret;
    };
}

Function22 function22_from_py_obj(py::object func) {
    return [func](Vector2 p) -> Vector2 {
        // TODO: GIL here seems inefficient...
        PyGILState_STATE state = PyGILState_Ensure();
        py::function f = py::reinterpret_borrow<py::function>(func);
        Vector2 ret = f(p).cast<Vector2>();
        PyGILState_Release(state);
        return ret;
    };
}

std::vector<Triangle> merge_mesh(const std::vector<Triangle> &a, const std::vector<Triangle> &b) {
    std::vector<Triangle> merged = a;
    merged.insert(merged.end(), b.begin(), b.end());
    return merged;
}


void export_visual(py::module &m) {
    DEFINE_VECTOR_OF_NAMED(RenderParticle, "RenderParticles");
    DEFINE_VECTOR_OF_NAMED(Triangle, "Triangles");

    m.def("function23_from_py_obj", function23_from_py_obj);
    m.def("function22_from_py_obj", function22_from_py_obj);
    m.def("generate_mesh", Mesh3D::generate);
    m.def("merge_mesh", merge_mesh);
    m.def("rasterize_render_particles", rasterize_render_particles);
    m.def("register_sdf", &AssetManager::insert_asset<SDF>);
    m.def("register_texture", &AssetManager::insert_asset<Texture>);
    m.def("register_surface_material", &AssetManager::insert_asset<SurfaceMaterial>);
    // TODO: these should registered by iterating over existing interfaces.
    m.def("create_mesh", std::make_shared<Mesh>);
    m.def("create_scene", std::make_shared<Scene>);


    py::class_<Texture, std::shared_ptr<Texture>>(m, "Texture")
            .def("initialize", &Texture::initialize);;

    py::class_<VolumeMaterial, std::shared_ptr<VolumeMaterial>>(m, "VolumeMaterial")
            .def("initialize", &VolumeMaterial::initialize);;

    py::class_<SurfaceMaterial, std::shared_ptr<SurfaceMaterial>>(m, "SurfaceMaterial")
            .def("initialize", static_cast<void (SurfaceMaterial::*)(const Config &)>(&SurfaceMaterial::initialize))
            .def("set_internal_material", &SurfaceMaterial::set_internal_material);

    py::class_<EnvironmentMap, std::shared_ptr<EnvironmentMap>>(m, "EnvironmentMap")
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
            .def("initialize",
                 static_cast<void (Camera::*)(const Config &config)>(&Camera::initialize));

    py::class_<ParticleRenderer, std::shared_ptr<ParticleRenderer>>(m, "ParticleRenderer")
            .def("initialize", &ParticleRenderer::initialize)
            .def("set_camera", &ParticleRenderer::set_camera)
            .def("render", &ParticleRenderer::render);

    py::class_<SDF, std::shared_ptr<SDF>>(m, "SDF")
            .def("initialize", &SDF::initialize)
            .def("eval", &SDF::eval);

    py::class_<Function22>(m, "Function22");
    py::class_<Function23>(m, "Function23");

}

TC_NAMESPACE_END

