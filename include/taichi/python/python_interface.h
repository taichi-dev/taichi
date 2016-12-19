#pragma once

#pragma warning(push)
#pragma warning(disable:4244)
#pragma warning(disable:4267)

#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#pragma warning(pop)

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarrayobject.h>

#include <vector>
#include <taichi/visual/camera.h>
#include <taichi/visual/renderer.h>
#include <taichi/visual/volume_material.h>
#include <taichi/visual/surface_material.h>
#include <taichi/visualization/particle_visualization.h>
#include <taichi/common/util.h>
#include <taichi/common/meta.h>
#include <taichi/dynamics/apic.h>
#include <taichi/dynamics/euler_smoke.h>
#include <taichi/dynamics/simulation3d.h>
#include <taichi/mpm2d/mpm.h>
#include <taichi/levelset/levelset2d.h>
#include <taichi/visualization/rgb.h>
#include <taichi/io/io.h>

#define EXPLICIT_GET_POINTER(T) namespace boost { template <> T const volatile * get_pointer<T const volatile >(T const volatile *c){return c;}}

EXPLICIT_GET_POINTER(taichi::MPMParticle);

EXPLICIT_GET_POINTER(taichi::EPParticle);

EXPLICIT_GET_POINTER(taichi::DPParticle);

EXPLICIT_GET_POINTER(taichi::Camera);

EXPLICIT_GET_POINTER(taichi::SurfaceMaterial);

EXPLICIT_GET_POINTER(taichi::VolumeMaterial);

EXPLICIT_GET_POINTER(taichi::Renderer);

EXPLICIT_GET_POINTER(taichi::Scene);

EXPLICIT_GET_POINTER(taichi::Mesh);

EXPLICIT_GET_POINTER(taichi::EnvironmentMap);

EXPLICIT_GET_POINTER(taichi::Texture);

EXPLICIT_GET_POINTER(taichi::ParticleRenderer);

EXPLICIT_GET_POINTER(taichi::Simulation3D);

TC_NAMESPACE_BEGIN

using namespace boost::python;
namespace py = boost::python;
using namespace std;

typedef std::vector<int> VectorInt;
typedef std::map<std::string, std::string> ConfigData;

Config config_from_py_dict(py::dict &c) {
	Config config;
	py::list keys = c.keys();
	for (int i = 0; i < len(keys); ++i) {
		py::object curArg = c[keys[i]];
		std::string key = py::extract<std::string>(keys[i]);
		std::string value = py::extract<std::string>(c[keys[i]]);
		config.set(key, value);
	}
	config.print_all();
	return config;
}

std::vector<float> make_range(float start, float end, float delta) {
	return std::vector<float> {start, end, delta};
}

std::string rasterize_levelset(const LevelSet2D &levelset, int width, int height) {
	std::string ret;
	for (auto &ind : Region2D(0, width, 0, height)) {
		float c = -levelset.sample((ind.i + 0.5f) / width * levelset.get_width(),
			(ind.j + 0.5f) / height * levelset.get_height());
		RGB rgb(c, c, c);
		rgb.append_to_string(ret);
	}
	return ret;
}

template<typename T>
void array2d_to_ndarray(T *arr, long long);

template<typename T>
void image_buffer_to_ndarray(T *arr, long long);

BOOST_PYTHON_MODULE(taichi_core) {
	Py_Initialize();
	//import_array();
	numeric::array::set_module_and_type("numpy", "ndarray");
	def("create_texture", create_instance<Texture>);
	def("register_texture", &AssetManager::insert_asset<Texture>);
	def("register_surface_material", &AssetManager::insert_asset<SurfaceMaterial>);
	def("create_simulation3d", create_instance<Simulation3D>);
	def("create_renderer", create_instance<Renderer>);
	def("create_camera", create_instance<Camera>);
	def("create_particle_renderer", create_instance<ParticleRenderer>);
	def("create_surface_material", create_instance<SurfaceMaterial>);
	def("create_volume_material", create_instance<VolumeMaterial>);
	def("create_environment_map", create_instance<EnvironmentMap>);
	def("create_mesh", std::make_shared<Mesh>);
	def("create_scene", std::make_shared<Scene>);
	def("config_from_dict", config_from_py_dict);
	def("rasterize_levelset", rasterize_levelset);
	def("points_inside_polygon", points_inside_polygon);
	def("points_inside_sphere", points_inside_sphere);
	def("make_range", make_range);
#define EXPORT_SIMULATOR(SIM) \
        class_<SIM>(#SIM) \
        .def("initialize", &SIM::initialize) \
        .def("step", &SIM::step) \
        .def("add_particle", &SIM::add_particle) \
        .def("get_current_time", &SIM::get_current_time) \
        .def("get_particles", &SIM::get_particles) \
        .def("set_levelset", &SIM::set_levelset) \
        .def("get_liquid_levelset", &SIM::get_liquid_levelset) \
        .def("get_density", &SIM::get_density) \
        .def("get_pressure", &SIM::get_pressure) \
        .def("add_source", &SIM::add_source) \
        ;
#define EXPORT_SIMULATOR_3D(SIM) \
        class_<SIM>(#SIM) \
        .def("initialize", &SIM::initialize) \
        .def("step", &SIM::step) \
        .def("get_current_time", &SIM::get_current_time) \
        .def("get_render_particles", &SIM::get_render_particles) \
        ;
	EXPORT_SIMULATOR_3D(Simulation3D);

#define EXPORT_MPM(SIM) \
    class_<SIM>(#SIM "Simulator") \
        .def("initialize", &SIM::initialize) \
        .def("step", &SIM::step) \
        .def("add_particle", static_cast<void (SIM::*)(std::shared_ptr<MPMParticle>)>(&SIM::add_particle)) \
        .def("get_current_time", &SIM::get_current_time) \
        .def("get_particles", &SIM::get_particles) \
        .def("set_levelset", &SIM::set_levelset) \
        .def("get_material_levelset", &SIM::get_material_levelset) \
        .def("add_ep_particle", static_cast<void (SIM::*)(EPParticle)>(&SIM::add_particle)) \
        .def("add_dp_particle", static_cast<void (SIM::*)(DPParticle)>(&SIM::add_particle)) \
        ;
	EXPORT_SIMULATOR(EulerFluid);
	EXPORT_SIMULATOR(EulerSmoke);
	EXPORT_SIMULATOR(FLIPFluid);

	EXPORT_MPM(MPM);

	class_<Config>("Config");

	class_<Vector2>("Vector2")
		.def_readwrite("x", &Vector2::x)
		.def_readwrite("y", &Vector2::y)
		.def(self * float())
		.def(float() * self)
		.def(self / float())
		.def(self + self)
		.def(self - self)
		.def(self * self)
		.def(self / self);

	class_<Vector3i>("Vector3i")
		.def_readwrite("x", &Vector3i::x)
		.def_readwrite("y", &Vector3i::y)
		.def_readwrite("z", &Vector3i::z)
		.def(self * int())
		.def(int() * self)
		.def(self / int())
		.def(self + self)
		.def(self - self)
		.def(self * self)
		.def(self / self);

	class_<Vector3>("Vector3")
		.def_readwrite("x", &Vector3::x)
		.def_readwrite("y", &Vector3::y)
		.def_readwrite("z", &Vector3::z)
		.def(self * float())
		.def(float() * self)
		.def(self / float())
		.def(self + self)
		.def(self - self)
		.def(self * self)
		.def(self / self);

	class_<Vector4>("Vector4")
		.def_readwrite("x", &Vector4::x)
		.def_readwrite("y", &Vector4::y)
		.def_readwrite("z", &Vector4::z)
		.def_readwrite("w", &Vector4::w)
		.def(self * float())
		.def(float() * self)
		.def(self / float())
		.def(self + self)
		.def(self - self)
		.def(self * self)
		.def(self / self);

	class_<Fluid::Particle>("FluidParticle", init<Vector2, Vector2>())
		.def_readwrite("position", &Fluid::Particle::position)
		.def_readwrite("velocity", &Fluid::Particle::velocity)
		.def_readwrite("color", &Fluid::Particle::color)
		.def_readwrite("temperature", &Fluid::Particle::temperature);
	class_<MPMParticle, std::shared_ptr<MPMParticle>>("MPMParticle")
		.def_readwrite("position", &MPMParticle::pos)
		.def_readwrite("velocity", &MPMParticle::v)
		.def_readwrite("color", &MPMParticle::color);
	class_<EPParticle, std::shared_ptr<EPParticle>, bases<MPMParticle>>("EPParticle")
		.def_readwrite("theta_c", &EPParticle::theta_c)
		.def_readwrite("theta_s", &EPParticle::theta_s)
		.def_readwrite("mu_0", &EPParticle::mu_0)
		.def_readwrite("lambda_0", &EPParticle::lambda_0)
		.def_readwrite("hardening", &EPParticle::hardening)
		.def_readwrite("mass", &EPParticle::mass)
		.def("set_compression", &EPParticle::set_compression);
	class_<DPParticle, std::shared_ptr<DPParticle>, bases<MPMParticle>>("DPParticle")
		.def_readwrite("h_0", &DPParticle::h_0)
		.def_readwrite("h_1", &DPParticle::h_1)
		.def_readwrite("h_2", &DPParticle::h_2)
		.def_readwrite("h_3", &DPParticle::h_3)
		.def_readwrite("mu_0", &DPParticle::mu_0)
		.def_readwrite("lambda_0", &DPParticle::lambda_0)
		.def_readwrite("alpha", &DPParticle::alpha)
		.def_readwrite("q", &DPParticle::q)
		.def_readwrite("mass", &DPParticle::mass)
		.def_readwrite("phi_f", &DPParticle::phi_f);

	class_<Array>("Array2DFloat")
		.def("to_ndarray", &array2d_to_ndarray<Array2D<float>>)
		.def("rasterize", &Array2D<float>::rasterize);

	class_<ImageBuffer<Vector3>>("RGBImageFloat", init<int, int, Vector3>())
		.def("get_width", &ImageBuffer<Vector3>::get_width)
		.def("get_height", &ImageBuffer<Vector3>::get_height)
		.def("to_ndarray", &image_buffer_to_ndarray<ImageBuffer<Vector3>>);

	class_<LevelSet2D>("LevelSet2D", init<int, int, Vector2>())
		.def("get", &LevelSet2D::get_copy)
		.def("set", static_cast<void (LevelSet2D::*)(int, int, const float &)>(&LevelSet2D::set))
		.def("add_sphere", &LevelSet2D::add_sphere)
		.def("add_polygon", &LevelSet2D::add_polygon)
		.def("get_gradient", &LevelSet2D::get_gradient)
		.def("rasterize", &LevelSet2D::rasterize)
		.def("sample", static_cast<float (LevelSet2D::*)(float, float) const>(&LevelSet2D::sample))
		.def("get_normalized_gradient", &LevelSet2D::get_normalized_gradient)
		.def("to_ndarray", &array2d_to_ndarray<LevelSet2D>)
		.def_readwrite("friction", &LevelSet2D::friction);

	typedef std::vector<Fluid::Particle> FluidParticles;
	class_<FluidParticles>("FluidParticles")
		.def(vector_indexing_suite<FluidParticles>());;
#define DEFINE_VECTOR_OF_NAMED(x, name) \
    class_<std::vector<x>>(name, init<>()) \
        .def(vector_indexing_suite<std::vector<x>, true>()) \
        .def("append", static_cast<void (std::vector<x>::*)(const x &)>(&std::vector<x>::push_back)) \
        .def("clear", &std::vector<x>::clear) \
        .def("write", &write_vector_to_disk<x>) \
        .def("read", &read_vector_from_disk<x>) \
    ;
#define DEFINE_VECTOR_OF(x) \
    class_<std::vector<x>>(#x "List", init<>()) \
        .def(vector_indexing_suite<std::vector<x>, true>()) \
        .def("append", static_cast<void (std::vector<x>::*)(const x &)>(&std::vector<x>::push_back)) \
        .def("clear", &std::vector<x>::clear) \
        .def("write", &write_vector_to_disk<x>) \
        .def("read", &read_vector_from_disk<x>) \
    ;

	DEFINE_VECTOR_OF(Vector2);
	DEFINE_VECTOR_OF_NAMED(RenderParticle, "RenderParticles");
	DEFINE_VECTOR_OF_NAMED(std::shared_ptr<MPMParticle>, "MPMParticles");
	DEFINE_VECTOR_OF(float);

	//class_<std::vector<Vector2>>("Vector2List", init<>())
	//	.def(vector_indexing_suite<std::vector<Vector2>>())
	//	.def("append", static_cast<void (std::vector<Vector2>::*)(const Vector2 &)>(&std::vector<Vector2>::push_back))
	//	.def("clear", &std::vector<Vector2>::clear)
	//;

	EXPORT_SIMULATOR(APICFluid);
	// class_<APICFluid, bases<Simulator>>("APICFluid");

	class_<Texture>("Texture")
		.def("initialize", &Texture::initialize);
	;

	class_<VolumeMaterial>("VolumeMaterial")
		.def("initialize", &VolumeMaterial::initialize);
	;

	class_<SurfaceMaterial>("SurfaceMaterial")
		.def("initialize", static_cast<void(SurfaceMaterial::*)(const Config &)>(&SurfaceMaterial::initialize))
		.def("set_internal_material", &SurfaceMaterial::set_internal_material)
		;

	class_<EnvironmentMap>("EnvironmentMap")
		.def("initialize", &EnvironmentMap::initialize)
		;

	class_<Mesh>("Mesh")
		.def("initialize", &Mesh::initialize)
		.def("set_material", &Mesh::set_material)
		.def("translate", &Mesh::translate)
		.def("scale", &Mesh::scale)
		.def("scale_s", &Mesh::scale_s)
		.def("rotate_angle_axis", &Mesh::rotate_angle_axis)
		.def("rotate_euler", &Mesh::rotate_euler)
		;

	class_<Scene>("Scene")
		//.def("initialize", &Scene::initialize)
		.def("finalize", &Scene::finalize)
		.def("add_mesh", &Scene::add_mesh)
		.def("set_atmosphere_material", &Scene::set_atmosphere_material)
		.def("set_environment_map", &Scene::set_environment_map)
		.def("set_camera", &Scene::set_camera)
		;

	// Renderers
	class_<Renderer>("Renderer")
		.def("initialize", &Renderer::initialize)
		.def("set_scene", &Renderer::set_scene)
		.def("render_stage", &Renderer::render_stage)
		.def("write_output", &Renderer::write_output)
		.def("get_output", &Renderer::get_output)
		;

	class_<Camera>("Camera")
		.def("initialize",
			static_cast<void (Camera::*)(const Config &config)>(&Camera::initialize));

	class_<ParticleRenderer>("ParticleRenderer")
		.def("initialize", &ParticleRenderer::initialize)
		.def("set_camera", &ParticleRenderer::set_camera)
		.def("render", &ParticleRenderer::render);
	register_ptr_to_python<std::shared_ptr<Renderer>>();
	register_ptr_to_python<std::shared_ptr<Camera>>();
	register_ptr_to_python<std::shared_ptr<SurfaceMaterial>>();
	register_ptr_to_python<std::shared_ptr<VolumeMaterial>>();
	register_ptr_to_python<std::shared_ptr<EnvironmentMap>>();
	register_ptr_to_python<std::shared_ptr<Mesh>>();
	register_ptr_to_python<std::shared_ptr<Scene>>();
	register_ptr_to_python<std::shared_ptr<Texture>>();
	register_ptr_to_python<std::shared_ptr<ParticleRenderer>>();
	register_ptr_to_python<std::shared_ptr<Simulation3D>>();
}

TC_NAMESPACE_END
