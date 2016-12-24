#include <taichi/python/export.h>

#include <taichi/visual/camera.h>
#include <taichi/visual/renderer.h>
#include <taichi/visual/volume_material.h>
#include <taichi/visual/surface_material.h>
#include <taichi/visualization/particle_visualization.h>

#include <taichi/geometry/geometry_factory.h>

using namespace boost::python;

EXPLICIT_GET_POINTER(taichi::Camera);

EXPLICIT_GET_POINTER(taichi::SurfaceMaterial);

EXPLICIT_GET_POINTER(taichi::VolumeMaterial);

EXPLICIT_GET_POINTER(taichi::Renderer);

EXPLICIT_GET_POINTER(taichi::Scene);

EXPLICIT_GET_POINTER(taichi::Mesh);

EXPLICIT_GET_POINTER(taichi::EnvironmentMap);

EXPLICIT_GET_POINTER(taichi::Texture);

EXPLICIT_GET_POINTER(taichi::ParticleRenderer);

TC_NAMESPACE_BEGIN

    std::function<Vector3(real, real)> surface_generator_from_py_obj(PyObject *func) {
        return [func](real u, real v) -> Vector3 {
            // TODO: GIL here seems inefficient...
            PyGILState_STATE state = PyGILState_Ensure();
            Vector3 ret = boost::python::call<Vector3>(func, u, v);
            PyGILState_Release(state);
            return ret;
        };
    }

    void export_visual() {
        def("surface_generator_from_py_obj", surface_generator_from_py_obj);
        def("generate_mesh", Mesh3D::generate);

        def("rasterize_render_particles", rasterize_render_particles);
        def("create_texture", create_instance<Texture>);
        def("register_texture", &AssetManager::insert_asset<Texture>);
        def("register_surface_material", &AssetManager::insert_asset<SurfaceMaterial>);
        def("create_renderer", create_instance<Renderer>);
        def("create_camera", create_instance<Camera>);
        def("create_particle_renderer", create_instance<ParticleRenderer>);
        def("create_surface_material", create_instance<SurfaceMaterial>);
        def("create_volume_material", create_instance<VolumeMaterial>);
        def("create_environment_map", create_instance<EnvironmentMap>);
        def("create_mesh", std::make_shared<Mesh>);
        def("create_scene", std::make_shared<Scene>);
        class_<ImageBuffer<Vector3 >>("RGBImageFloat", init<int, int, Vector3>())
                .def("get_width", &ImageBuffer<Vector3>::get_width)
                .def("get_height", &ImageBuffer<Vector3>::get_height)
                .def("to_ndarray", &image_buffer_to_ndarray < ImageBuffer<Vector3>>);
        class_<Texture>("Texture")
                .def("initialize", &Texture::initialize);;

        class_<VolumeMaterial>("VolumeMaterial")
                .def("initialize", &VolumeMaterial::initialize);;

        class_<SurfaceMaterial>("SurfaceMaterial")
                .def("initialize", static_cast<void (SurfaceMaterial::*)(const Config &)>(&SurfaceMaterial::initialize))
                .def("set_internal_material", &SurfaceMaterial::set_internal_material);

        class_<EnvironmentMap>("EnvironmentMap")
                .def("initialize", &EnvironmentMap::initialize);

        class_<Mesh>("Mesh")
                .def("initialize", &Mesh::initialize)
                .def("set_material", &Mesh::set_material)
                .def_readwrite("transform", &Mesh::transform)
                .def("translate", &Mesh::translate)
                .def("scale", &Mesh::scale)
                .def("scale_s", &Mesh::scale_s)
                .def("rotate_angle_axis", &Mesh::rotate_angle_axis)
                .def("rotate_euler", &Mesh::rotate_euler);

        class_<Scene>("Scene")
                //.def("initialize", &Scene::initialize)
                .def("finalize", &Scene::finalize)
                .def("add_mesh", &Scene::add_mesh)
                .def("set_atmosphere_material", &Scene::set_atmosphere_material)
                .def("set_environment_map", &Scene::set_environment_map)
                .def("set_camera", &Scene::set_camera);

        // Renderers
        class_<Renderer>("Renderer")
                .def("initialize", &Renderer::initialize)
                .def("set_scene", &Renderer::set_scene)
                .def("render_stage", &Renderer::render_stage)
                .def("write_output", &Renderer::write_output)
                .def("get_output", &Renderer::get_output);

        class_<Camera>("Camera")
                .def("initialize",
                     static_cast<void (Camera::*)(const Config &config)>(&Camera::initialize));

        class_<ParticleRenderer>("ParticleRenderer")
                .def("initialize", &ParticleRenderer::initialize)
                .def("set_camera", &ParticleRenderer::set_camera)
                .def("render", &ParticleRenderer::render);

        class_<Mesh3D::SurfaceGenerator>("surface_generator");

        DEFINE_VECTOR_OF_NAMED(RenderParticle, "RenderParticles");

        register_ptr_to_python<std::shared_ptr<Renderer>>();
        register_ptr_to_python<std::shared_ptr<Camera>>();
        register_ptr_to_python<std::shared_ptr<SurfaceMaterial>>();
        register_ptr_to_python<std::shared_ptr<VolumeMaterial>>();
        register_ptr_to_python<std::shared_ptr<EnvironmentMap>>();
        register_ptr_to_python<std::shared_ptr<Mesh>>();
        register_ptr_to_python<std::shared_ptr<Scene>>();
        register_ptr_to_python<std::shared_ptr<Texture>>();
        register_ptr_to_python<std::shared_ptr<ParticleRenderer>>();
    }

TC_NAMESPACE_END

