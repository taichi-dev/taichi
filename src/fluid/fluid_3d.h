#pragma once

#include <memory.h>
#include <string>
#include "visualization/image_buffer.h"
#include "common/config.h"
#include "common/interface.h"
#include "math/array_3d.h"
#include "pressure_solver.h"
#include "fluid/simulation3d.h"

TC_NAMESPACE_BEGIN

class Tracker3D {
public:
	Vector3 position;
	Vector3 color;
	Tracker3D() {}
	Tracker3D(const Vector3 &position, const Vector3 &color) : position(position), color(color) {}
};

class Smoke3D : public Simulation3D {
	typedef Array3D<float> Array;
public:
	Array u, v, w, rho, t, pressure, last_pressure;
	Vector3i res;
	float smoke_alpha, smoke_beta;
	float temperature_decay;
	float pressure_tolerance;
	float density_scaling;
	Vector3 initial_speed;
	float tracker_generation;
	float perturbation;
	std::vector<Tracker3D> trackers;
	std::shared_ptr<PressureSolver3D> pressure_solver;

	Smoke3D() {}

	void remove_outside_trackers();

	void initialize(const Config &config);

	void project();

	void confine_vorticity(float delta_t);

	void advect(float delta_t);

	void move_trackers(float delta_t);

	void step(float delta_t);

	virtual void show(ImageBuffer<Vector3> &buffer);

	void advect(Array &attr, float delta_t);

	void apply_boundary_condition();

	static Vector3 sample_velocity(const Array &u, const Array &v, const Array &w, const Vector3 &pos);

	Vector3 sample_velocity(const Vector3 &pos) const;

	std::vector<RenderParticle> get_render_particles() const;
};

TC_NAMESPACE_END
