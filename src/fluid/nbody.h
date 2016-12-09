#pragma once

#include "common/meta.h"
#include "common/interface.h"
#include "particle_visualization.h"

TC_NAMESPACE_BEGIN

class NBody : public Simulator {
protected:
	float current_t;
public:
	float get_current_time() {
		return current_t;
	}
	virtual void initialize(const Config &config) override {

	}
	std::vector<RenderParticle> get_render_particles() const;
};

TC_NAMESPACE_END

#pragma once
