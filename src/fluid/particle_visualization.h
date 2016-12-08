#pragma once

#include "common/utils.h"
#include "visualization/image_buffer.h"
#include "common/config.h"
#include "renderer/camera.h"
#include "common/meta.h"

TC_NAMESPACE_BEGIN

struct RenderParticle {
	Vector3 position;
	Vector4 color;
	RenderParticle() {}
	RenderParticle(const Vector3 &position, const Vector4 &color) : position(position), color(color) {}
	bool operator == (const RenderParticle &p) const {
		// For boost::python vector_indexing_suite
		return false;
	}
};

class ParticleRenderer {
protected:
	std::shared_ptr<Camera> camera;

public:
	void set_camera(std::shared_ptr<Camera> camera) {
		this->camera = camera;
	}
	virtual void initialize(const Config &config) {};
	virtual void render(ImageBuffer<Vector3> &buffer, const std::vector<RenderParticle> &particles) const {}
};


TC_INTERFACE(ParticleRenderer)

TC_NAMESPACE_END

