#include <taichi/visual/renderer.h>

TC_NAMESPACE_BEGIN

void Renderer::initialize(const Config &config) {
	this->ray_intersection = create_instance<RayIntersection>(config.get("ray_intersection", "embree"));
	sg = std::make_shared<SceneGeometry>(scene, ray_intersection);
	this->min_path_length = config.get_int("min_path_length");
	this->max_path_length = config.get_int("max_path_length");
	this->num_threads = config.get("num_threads", 1);
	assert_info(min_path_length <= max_path_length, "min_path_length > max_path_length");
}

void Renderer::set_scene(std::shared_ptr<Scene> scene) {
	this->scene = scene;
	this->camera = scene->camera;
	this->width = camera->get_width();
	this->height = camera->get_height();
}

void Renderer::write_output(std::string fn) {
	auto tmp = get_output();
	Vector3 sum(0.0f);
	for (auto p : tmp) {
		sum += p;
	}
	auto scale = luminance(sum) / luminance(Vector3(1.0f)) / tmp.get_width() / tmp.get_height() / 0.18f;
	for (auto ind : tmp.get_region()) {
		for (int i = 0; i < 3; i++) {
			tmp[ind][i] = std::pow(clamp(tmp[ind][i] / scale, 0.0f, 1.0f), 1 / 2.2f);
		}
	}
	tmp.write(fn);
}

TC_INTERFACE_DEF(Renderer, "renderer");

TC_NAMESPACE_END

