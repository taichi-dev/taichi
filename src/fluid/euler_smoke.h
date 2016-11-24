#pragma once

#include "euler_fluid.h"

TC_NAMESPACE_BEGIN

class EulerSmoke : public EulerFluid {
protected:
	//int advection_order;


	//virtual void advect(float delta_t);
	virtual void emit(float delta_t);

	virtual void substep(float delta_t);

	Array temperature;
	float buoyancy_alpha;
	float buoyancy_beta;

public:
	EulerSmoke() {}

	virtual void apply_external_forces(float delta_t);

	virtual void initialize(const Config &config);

};

TC_NAMESPACE_END

