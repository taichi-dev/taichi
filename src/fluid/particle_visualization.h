#pragma once

#include "common/utils.h"
#include "visualization/image_buffer.h"
#include "common/config.h"

TC_NAMESPACE_BEGIN

class ParticleShadowMapRenderer {
private:
	Vector3 light_direction;
	real rotate_z;
	Vector3 center;
	int shadow_map_resolution;
	real shadowing;
	real alpha;
public:
	struct Particle {
		Vector3 position;
		Vector3 color;
		Particle() {}
		Particle(const Vector3 &position, const Vector3 &color) : position(position), color(color) {}
	};

	ParticleShadowMapRenderer() {
	}

	void initialize(const Config &config) {
		alpha = config.get_real("alpha");
		shadowing = config.get_real("shadowing");
		shadow_map_resolution = config.get_int("shadow_map_resolution");
		light_direction = config.get_vec3("light_direction");
		//rotate_z = config.get_real("rotate_z");
		rotate_z = 30.0f;
		center = config.get_vec3("center");
	}

	void render(ImageBuffer<Vector3> &buffer, const std::vector<Particle> particles);
};

TC_NAMESPACE_END

