#pragma once

#include <memory.h>
#include <string>
#include "visualization/image_buffer.h"
#include "common/config.h"
#include "common/interface.h"
#include "math/array_3d.h"
#include "pressure_solver.h"
#include "particle_visualization.h"

TC_NAMESPACE_BEGIN
class ParticleShadowMapRenderer;

class Fluid3D : public Simulator {
protected:
	float current_t;
public:
	virtual void show(ImageBuffer<Vector3> &buffer) {};
	virtual ImageBuffer<Vector3> get_visualization(int width, int height) { return ImageBuffer<Vector3>(1, 1); };
	float get_current_time() {
		return current_t;
	}

};

class Tracker3D {
public:
	Vector3 position;
	Vector3 color;
	Tracker3D() {}
	Tracker3D(const Vector3 &position, const Vector3 &color) : position(position), color(color) {}
};

class Smoke3D : public Fluid3D {
	typedef Array3D<float> Array;
public:
	Array u, v, w, rho, t, pressure, last_pressure;
	int width, height, depth;
	float smoke_alpha, smoke_beta;
	float temperature_decay;
	float pressure_tolerance;
	float density_scaling;
	Vector3 initial_speed;
	bool show_trackers;
	float tracker_generation;
	float perturbation;
	float viewport_rotation;
	std::vector<Tracker3D> trackers;
	std::shared_ptr<PressureSolver3D> pressure_solver;
	std::shared_ptr<ParticleShadowMapRenderer> particle_renderer;

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

std::shared_ptr<Fluid3D> create_fluid_3d(std::string name, const Config &config);

TC_NAMESPACE_END
