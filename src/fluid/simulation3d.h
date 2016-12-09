#pragma once

#include "common/meta.h"
#include "particle_visualization.h"
#include <vector>

TC_NAMESPACE_BEGIN

class Simulation3D {
protected:
	real current_t;
public:
	Simulation3D() {}
	virtual float get_current_time() const {
		return current_t;
	}
	virtual void initialize(const Config &config) {}
	virtual void step(real t) {
		error("no impl");
	}
	virtual std::vector<RenderParticle> get_render_particles() const {
		error("no impl");
		return std::vector<RenderParticle>();
	}
};

TC_INTERFACE(Simulation3D);

TC_NAMESPACE_END
