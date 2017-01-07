#include <taichi/python/export.h>

#include <taichi/visual/camera.h>
#include <taichi/visual/renderer.h>
#include <taichi/visual/volume_material.h>
#include <taichi/visual/surface_material.h>
#include <taichi/visualization/particle_visualization.h>

#include <taichi/geometry/factory.h>

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

Function23 function23_from_py_obj(PyObject *func) {
    return [func](Vector2 p) -> Vector3 {
        // TODO: GIL here seems inefficient...
        PyGILState_STATE state = PyGILState_Ensure();
        Vector3 ret = boost::python::call<Vector3>(func, p);
        PyGILState_Release(state);
        return ret;
    };
}

Function22 function22_from_py_obj(PyObject *func) {
    return [func](Vector2 p) -> Vector2 {
        // TODO: GIL here seems inefficient...
        PyGILState_STATE state = PyGILState_Ensure();
        Vector2 ret = boost::python::call<Vector2>(func, p);
        PyGILState_Release(state);
        return ret;
    };
}

std::vector<Triangle> merge_mesh(const std::vector<Triangle> &a, const std::vector<Triangle> &b) {
    std::vector<Triangle> merged = a;
    merged.insert(merged.end(), b.begin(), b.end());
    return merged;
}

template<typename T, int ret>
int return_constant(T *) { return ret; }


    template<typename T, int channels>
    void ndarray_to_image_buffer(T *arr, long long input, int width, int height) // 'input' is actually a pointer...
    {
        arr->initialize(width, height);
        for (auto &ind : arr->get_region()) {
            for (int i = 0; i < channels; i++) {
                (*arr)[ind][i] = reinterpret_cast<float *>(input)[ind.i * channels + ind.j * width * channels + i];
            }
        }
    }

void export_visual() {
    def("function23_from_py_obj", function23_from_py_obj);
    def("function22_from_py_obj", function22_from_py_obj);
    def("generate_mesh", Mesh3D::generate);
    def("merge_mesh", merge_mesh);
    def("rasterize_render_particles", rasterize_render_particles);
    def("register_texture", &AssetManager::insert_asset<Texture>);
    def("register_surface_material", &AssetManager::insert_asset<SurfaceMaterial>);
    // TODO: these should registered by iterating over existing interfaces.
    def("create_texture", create_instance<Texture>);
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
            .def("get_channels", &return_constant<ImageBuffer<Vector3>, 3>)
            .def("from_array2d", &ImageBuffer<Vector3>::from_array2d)
            .def("from_ndarray", &ndarray_to_image_buffer<ImageBuffer<Vector3>, 3>)
            .def("read", &ImageBuffer<Vector3>::load)
            .def("write", &ImageBuffer<Vector3>::write)
            .def("to_ndarray", &image_buffer_to_ndarray<ImageBuffer<Vector3>, 3>);
    class_<ImageBuffer<Vector4 >>("RGBAImageFloat", init<int, int, Vector4>())
        .def("get_width", &ImageBuffer<Vector4>::get_width)
        .def("get_height", &ImageBuffer<Vector4>::get_height)
        .def("get_channels", &return_constant<ImageBuffer<Vector4>, 4>)
        .def("write", &ImageBuffer<Vector4>::write)
        .def("from_array2d", &ImageBuffer<Vector4>::from_array2d)
        .def("from_ndarray", &ndarray_to_image_buffer<ImageBuffer<Vector4>, 4>)
        .def("to_ndarray", &image_buffer_to_ndarray<ImageBuffer<Vector4>, 4>);
    class_<Texture>("Texture")
        .def("initialize", &Texture::initialize);;

    class_<VolumeMaterial>("VolumeMaterial")
        .def("initialize", &VolumeMaterial::initialize);;

    class_<SurfaceMaterial>("SurfaceMaterial")
        .def("initialize", static_cast<void (SurfaceMaterial::*)(const Config &)>(&SurfaceMaterial::initialize))
        .def("set_internal_material", &SurfaceMaterial::set_internal_material);

    class_<EnvironmentMap>("EnvironmentMap")
        .def("initialize", &EnvironmentMap::initialize)
        .def("set_transform", &EnvironmentMap::set_transform);

    class_<Mesh>("Mesh")
        .def("initialize", &Mesh::initialize)
        .def("set_untransformed_triangles", &Mesh::set_untransformed_triangles)
        .def("set_material", &Mesh::set_material)
        .def_readwrite("transform", &Mesh::transform);

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

    class_<Function22>("Function22");
    class_<Function23>("Function23");

    DEFINE_VECTOR_OF_NAMED(RenderParticle, "RenderParticles");
    DEFINE_VECTOR_OF_NAMED(Triangle, "Triangles");

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

