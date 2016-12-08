#include "fluid_3d.h"
#include "common/utils.h"
#include "math/array_3d.h"
#include "math/array_2d.h"
#include "pressure_solver.h"
#include "particle_visualization.h"
#include "system/timer.h"

TC_NAMESPACE_BEGIN
const static Vector3i offsets[]{
		Vector3i(1, 0, 0), Vector3i(-1, 0, 0),
		Vector3i(0, 1, 0), Vector3i(0, -1, 0),
		Vector3i(0, 0, 1), Vector3i(0, 0, -1)
};

void Smoke3D::project() {
	Array divergence(width, height, depth, 0.0f);
	for (auto &ind : u.get_region()) {
		if (0 < ind.i)
			divergence[ind + Vector3i(-1, 0, 0)] += u[ind];
		if (ind.i < width)
			divergence[ind] -= u[ind];
	}
	for (auto &ind : v.get_region()) {
		if (0 < ind.j)
			divergence[ind + Vector3i(0, -1, 0)] += v[ind];
		if (ind.j < height)
			divergence[ind] -= v[ind];
	}
	for (auto &ind : w.get_region()) {
		if (0 < ind.k)
			divergence[ind + Vector3i(0, 0, -1)] += w[ind];
		if (ind.k < depth)
			divergence[ind] -= w[ind];
	}
	pressure = 0;
	pressure_solver->run(divergence, pressure, pressure_tolerance);
	for (auto &ind : pressure.get_region()) {
		u[ind] += pressure[ind];
		u[ind + Vector3i(1, 0, 0)] -= pressure[ind];
		v[ind] += pressure[ind];
		v[ind + Vector3i(0, 1, 0)] -= pressure[ind];
		w[ind] += pressure[ind];
		w[ind + Vector3i(0, 0, 1)] -= pressure[ind];
	}
	last_pressure = pressure;
}

void Smoke3D::initialize(const Config &config) {
	width = config.get("simulation_width", 32);
	height = config.get("simulation_height", 32);
	depth = config.get("simulation_depth", 32);
	smoke_alpha = config.get("smoke_alpha", 0.0f);
	smoke_beta = config.get("smoke_beta", 0.0f);
	temperature_decay = config.get("temperature_decay", 0.0f);
	pressure_tolerance = config.get("pressure_tolerance", 0.0f);
	density_scaling = config.get("density_scaling", 1.0f);
	initial_speed = config.get("initial_speed", Vector3(0, 0, 0));
	tracker_generation = config.get("tracker_generation", 100.0f);

	// TODO: refactor here

	show_trackers = config.get("show_trackers", true);
	perturbation = config.get("perturbation", 0.0f);
	viewport_rotation = config.get("viewport_rotation", 0.0f);
	Config solver_config;
	solver_config.set("width", width).set("height", height).set("depth", depth);
	pressure_solver = create_pressure_solver_3d(config.get_string("pressure_solver"), solver_config);
	u = Array(width + 1, height, depth, 0.0f, Vector3(0.0f, 0.5f, 0.5f));
	v = Array(width, height + 1, depth, 0.0f, Vector3(0.5f, 0.0f, 0.5f));
	w = Array(width, height, depth + 1, 0.0f, Vector3(0.5f, 0.5f, 0.0f));
	rho = Array(width, height, depth, 0.0f);
	pressure = Array(width, height, depth, 0.0f);
	last_pressure = Array(width, height, depth, 0.0f);
	t = Array(width, height, depth, config.get("initial_t", 0.0f));
	current_t = 0.0f;
}

Vector3 hsv2rgb(Vector3 hsv) {
	float h = hsv.x;
	float s = hsv.y;
	float v = hsv.z;
	int j = (int)floor(h * 6);
	float f = h * 6 - j;
	float p = v * (1 - s);
	float q = v * (1 - f * s);
	float t = v * (1 - (1 - f) * s);
	float r, g, b;
	switch (j % 6) {
	case 0: r = v, g = t, b = p; break;
	case 1: r = q, g = v, b = p; break;
	case 2: r = p, g = v, b = t; break;
	case 3: r = p, g = q, b = v; break;
	case 4: r = t, g = p, b = v; break;
	case 5: r = v, g = p, b = q; break;
	}
	return Vector3(r, g, b);
}

std::vector<RenderParticle> Smoke3D::get_render_particles() const {
	using Particle = RenderParticle;
	std::vector<Particle> render_particles;
	render_particles.reserve(trackers.size());
	Vector3 center(width / 2.0f, height / 2.0f, depth / 2.0f);
	for (auto p : trackers) {
		render_particles.push_back(Particle(p.position - center, Vector4(p.color.x, p.color.y, p.color.z, 1.0f)));
	}
	return render_particles;
}

void Smoke3D::show(ImageBuffer<Vector3> &buffer) {
	buffer.reset(Vector3(0));
	int half_width = buffer.get_width() / 2, half_height = buffer.get_height() / 2;
	for (int i = 0; i < half_width; i++) {
		for (int j = 0; j < buffer.get_height(); j++) {
			float rho_sum = 0.0f;
			float t_sum = 0.0f;
			for (int k = 0; k < depth; k++) {
				float x = (i + 0.5f) / (float)half_width * width;
				float y = (j + 0.5f) / (float)buffer.get_height() * height;
				float z = k + 0.5f;
				rho_sum += rho.sample(x, y, z);
				t_sum += t.sample(x, y, z);
			}
			rho_sum *= density_scaling;
			t_sum = min(1.0f, t_sum / depth);
			rho_sum = min(1.0f, rho_sum / depth);
			buffer[i][j] = Vector3(t_sum, rho_sum * 0.3f, rho_sum * 0.8f);
		}
	}
	for (int i = 0; i < half_width; i++) {
		for (int j = 0; j < half_height; j++) {
			float rho_sum = 0.0f;
			float t_sum = 0.0f;
			for (int k = 0; k < depth; k++) {
				float x = (i + 0.5f) / (float)half_width * width;
				float y = k + 0.5f;
				float z = (j + 0.5f) / (float)half_height * depth;
				rho_sum += rho.sample(x, y, z);
				t_sum += t.sample(x, y, z);
			}
			rho_sum *= density_scaling;
			t_sum = min(1.0f, t_sum / depth);
			rho_sum = min(1.0f, rho_sum / depth);
			buffer[half_width + i][j] = Vector3(t_sum, rho_sum * 0.3f, rho_sum * 0.8f);
		}
	}
}

void Smoke3D::move_trackers(float delta_t) {
	for (auto &tracker : trackers) {
		auto velocity = sample_velocity(tracker.position);
		tracker.position += sample_velocity(tracker.position + 0.5f * delta_t * velocity) * delta_t;
	}
}

void Smoke3D::step(float delta_t) {
	{
		Time::Timer _("Adding source");
		for (auto &ind : rho.get_region()) {
			if (length(ind.get_pos() - Vector3(width / 2.0f, height * 0.1f, depth / 2.0f)) < height * 0.05f) {
				rho[ind] = 1.0f;
				t[ind] = 1.0f;
				u[ind] = initial_speed.x;
				v[ind] = initial_speed.y;
				w[ind] = initial_speed.z;
				u[ind] += perturbation * (rand() - 0.5f);
				w[ind] += perturbation * (rand() - 0.5f);
				for (int i = 0; i < delta_t * tracker_generation; i++) {
					Vector3 position = ind.get_pos() - Vector3(0.5f) + Vector3(rand(), rand(), rand());
					float h = get_current_time() * 0.3f;
					h -= floor(h);
					Vector3 color = hsv2rgb(Vector3(h, 0.5f, 1.0f));
					trackers.push_back(Tracker3D(position, color));
				}
			}
		}
		for (auto &ind : v.get_region()) {
			if (ind.j < height) {
				v[ind] += (-smoke_alpha * rho[ind] + smoke_beta * t[ind]) * delta_t;
			}
		}
		float t_decay = exp(-delta_t * temperature_decay);
		for (auto &ind : t.get_region()) {
			t[ind] *= t_decay;
		}
	}
	apply_boundary_condition();
	TIME(project());
	apply_boundary_condition();
	TIME(move_trackers(delta_t));
	TIME(remove_outside_trackers());
	TIME(advect(delta_t));
	apply_boundary_condition();
	current_t += delta_t;
}

void Smoke3D::remove_outside_trackers() {
	std::vector<Tracker3D> all_trackers = trackers;
	trackers.clear();
	for (auto &tracker : all_trackers) {
		Vector3 p = tracker.position;
		if (0 <= p.x && p.x <= width && 0 <= p.y && p.y <= height && 0 <= p.z && p.z <= depth) {
			trackers.push_back(tracker);
		}
	}
}

Vector3 Smoke3D::sample_velocity(const Array &u, const Array &v, const Array &w, const Vector3 &pos) {
	return Vector3(u.sample(pos), v.sample(pos), w.sample(pos));
}

Vector3 Smoke3D::sample_velocity(const Vector3 &pos) const {
	return sample_velocity(u, v, w, pos);
}

void Smoke3D::advect(Array &attr, float delta_t) {
	auto new_attr = attr.same_shape(0);
	for (auto &ind : new_attr.get_region()) {
		auto old_position = ind.get_pos() - delta_t * sample_velocity(ind.get_pos());
		new_attr[ind] = attr.sample(old_position);
	}
	attr = new_attr;
}

void Smoke3D::apply_boundary_condition() {
	return;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			w[i][j][0] = w[i][j][depth - 1] = 0;
		}
	}
	for (int i = 0; i < width; i++) {
		for (int k = 0; k < depth; k++) {
			v[i][0][k] = v[i][height - 1][k] = 0;
		}
	}
	for (int j = 0; j < height; j++) {
		for (int k = 0; k < depth; k++) {
			u[0][j][k] = u[width - 1][j][k] = 0;
		}
	}
}

void Smoke3D::advect(float delta_t) {
	advect(rho, delta_t);
	advect(t, delta_t);
	advect(u, delta_t);
	advect(v, delta_t);
	advect(w, delta_t);
}

void Smoke3D::confine_vorticity(float delta_t) {

}

std::shared_ptr<Fluid3D> create_fluid_3d(std::string name, const Config &config) {
	auto fluid_3d = std::make_shared<Smoke3D>();
	fluid_3d->initialize(config);
	return fluid_3d;
}

TC_NAMESPACE_END
