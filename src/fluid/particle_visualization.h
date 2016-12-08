#pragma once

#include "common/utils.h"
#include "visualization/image_buffer.h"
#include "common/config.h"
#include "renderer/camera.h"

TC_NAMESPACE_BEGIN

class ParticleShadowMapRenderer {
private:
	Vector3 light_direction;
	real shadow_map_resolution;
	Matrix3 light_transform;
	std::shared_ptr<Camera> camera;
	real ambient_light;
	real shadowing;
public:
	struct Particle {
		Vector3 position;
		Vector4 color;
		Particle() {}
		Particle(const Vector3 &position, const Vector4 &color) : position(position), color(color) {}
	};

	void set_camera(std::shared_ptr<Camera> camera) {
		this->camera = camera;
	}

	ParticleShadowMapRenderer() {}

	void initialize(const Config &config);

	void render(ImageBuffer<Vector3> &buffer, const std::vector<Particle> particles);
};

TC_NAMESPACE_END

