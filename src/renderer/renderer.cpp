#include "renderer.h"

TC_NAMESPACE_BEGIN

    void Renderer::initialize(const Config &config) {
        this->ray_intersection = create_instance<RayIntersection>(config.get("ray_intersection", "embree"));
        this->width = config.get_int("width");
        this->height = config.get_int("height");
        this->min_path_length = config.get_int("min_path_length");
        this->max_path_length = config.get_int("max_path_length");
    }

    void Renderer::set_scene(std::shared_ptr<Scene> scene) {
        this->scene = scene;
        sg = std::make_shared<SceneGeometry>(scene, ray_intersection);
		if (scene->camera) // TODO: standalone camera specification
			this->camera = scene->camera;
    }
	
	void Renderer::write_output(std::string fn) {
		get_output().write(fn);
	}

    TC_INTERFACE_DEF(Renderer, "renderer");

TC_NAMESPACE_END

