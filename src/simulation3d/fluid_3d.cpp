#include "fluid_3d.h"
#include <taichi/common/util.h>
#include <taichi/math/array_3d.h>
#include <taichi/math/array_2d.h>
#include <taichi/dynamics/pressure_solver3d.h>
#include <taichi/visualization/particle_visualization.h>
#include <taichi/system/timer.h>

TC_NAMESPACE_BEGIN
const static Vector3i offsets[]{
		Vector3i(1, 0, 0), Vector3i(-1, 0, 0),
		Vector3i(0, 1, 0), Vector3i(0, -1, 0),
		Vector3i(0, 0, 1), Vector3i(0, 0, -1)
};

void Smoke3D::project() {
	Array divergence(res[0], res[1], res[2], 0.0f);
	for (auto &ind : u.get_region()) {
		if (0 < ind.i)
			divergence[ind + Vector3i(-1, 0, 0)] += u[ind];
		if (ind.i < res[0])
			divergence[ind] -= u[ind];
	}
	for (auto &ind : v.get_region()) {
		if (0 < ind.j)
			divergence[ind + Vector3i(0, -1, 0)] += v[ind];
		if (ind.j < res[1])
			divergence[ind] -= v[ind];
	}
	for (auto &ind : w.get_region()) {
		if (0 < ind.k)
			divergence[ind + Vector3i(0, 0, -1)] += w[ind];
		if (ind.k < res[2])
			divergence[ind] -= w[ind];
	}
	pressure = 0;
    pressure_solver->set_boundary_condition(boundary_condition);
    for (auto &ind : boundary_condition.get_region()) {
        if (boundary_condition[ind] != PressureSolver3D::INTERIOR) {
            divergence[ind] = 0.0f;
        }
    }
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
	Simulation3D::initialize(config);
	res = config.get_vec3i("resolution");
	smoke_alpha = config.get("smoke_alpha", 0.0f);
	smoke_beta = config.get("smoke_beta", 0.0f);
	temperature_decay = config.get("temperature_decay", 0.0f);
	pressure_tolerance = config.get("pressure_tolerance", 0.0f);
	density_scaling = config.get("density_scaling", 1.0f);
	initial_speed = config.get("initial_speed", Vector3(0, 0, 0));
	tracker_generation = config.get("tracker_generation", 100.0f);
	num_threads = config.get_int("num_threads");
    std::string padding;
    open_boundary = config.get_bool("open_boundary");
    if (open_boundary) {
        padding = "dirichlet";
    } else {
        padding = "neumann";
    };

	perturbation = config.get("perturbation", 0.0f);
	Config solver_config;
	solver_config.set("width", res[0]).set("height", res[1]).set("depth", res[2])
		.set("num_threads", num_threads).set("padding", padding);
	pressure_solver = create_initialized_instance<PressureSolver3D>(config.get_string("pressure_solver"), solver_config);
	u = Array(res[0] + 1, res[1], res[2], 0.0f, Vector3(0.0f, 0.5f, 0.5f));
	v = Array(res[0], res[1] + 1, res[2], 0.0f, Vector3(0.5f, 0.0f, 0.5f));
	w = Array(res[0], res[1], res[2] + 1, 0.0f, Vector3(0.5f, 0.5f, 0.0f));
	rho = Array(res[0], res[1], res[2], 0.0f);
	pressure = Array(res[0], res[1], res[2], 0.0f);
	last_pressure = Array(res[0], res[1], res[2], 0.0f);
	t = Array(res[0], res[1], res[2], config.get("initial_t", 0.0f));
	current_t = 0.0f;
    boundary_condition = PressureSolver3D::BCArray(res);
    for (auto &ind : boundary_condition.get_region()) {
        Vector3 d = ind.get_pos() - Vector3(res) * 0.5f;
        if (length(d) * 4 < res[0] || ind.i == 0 || ind.i == res[0] - 1 || ind.j == 0
                || ind.k == 0 || ind.k == res[2] - 1) {
            boundary_condition[ind] = PressureSolver3D::NEUMANN;
        }
    }
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
	Vector3 center(res[0] / 2.0f, res[1] / 2.0f, res[2] / 2.0f);
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
			for (int k = 0; k < res[2]; k++) {
				float x = (i + 0.5f) / (float)half_width * res[0];
				float y = (j + 0.5f) / (float)buffer.get_height() * res[1];
				float z = k + 0.5f;
				rho_sum += rho.sample(x, y, z);
				t_sum += t.sample(x, y, z);
			}
			rho_sum *= density_scaling;
			t_sum = std::min(1.0f, t_sum / res[2]);
			rho_sum = std::min(1.0f, rho_sum / res[2]);
			buffer[i][j] = Vector3(t_sum, rho_sum * 0.3f, rho_sum * 0.8f);
		}
	}
	for (int i = 0; i < half_width; i++) {
		for (int j = 0; j < half_height; j++) {
			float rho_sum = 0.0f;
			float t_sum = 0.0f;
			for (int k = 0; k < res[2]; k++) {
				float x = (i + 0.5f) / (float)half_width * res[0];
				float y = k + 0.5f;
				float z = (j + 0.5f) / (float)half_height * res[2];
				rho_sum += rho.sample(x, y, z);
				t_sum += t.sample(x, y, z);
			}
			rho_sum *= density_scaling;
			t_sum = std::min(1.0f, t_sum / res[2]);
			rho_sum = std::min(1.0f, rho_sum / res[2]);
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
			if (length(ind.get_pos() - Vector3(res[0] / 2.0f, res[1] * 0.1f, res[2] / 2.0f)) < res[1] * 0.05f) {
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
			if (ind.j < res[1]) {
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
		if (0 <= p.x && p.x <= res[0] && 0 <= p.y && p.y <= res[1] && 0 <= p.z && p.z <= res[2]) {
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
    for (auto &ind : boundary_condition.get_region()) {
        if (boundary_condition[ind] == PressureSolver3D::NEUMANN) {
            u[ind] = 0;
            u[ind + Vector3(1, 0, 0)] = 0;
            v[ind] = 0;
            v[ind + Vector3(0, 1, 0)] = 0;
            w[ind] = 0;
            w[ind + Vector3(0, 0, 1)] = 0;
        }
    }
    if (!open_boundary) {
        for (int i = 0; i < res[0]; i++) {
            for (int j = 0; j < res[1]; j++) {
                w[i][j][0] = w[i][j][res[2] - 1] = 0;
            }
        }
        for (int i = 0; i < res[0]; i++) {
            for (int k = 0; k < res[2]; k++) {
                v[i][0][k] = v[i][res[1] - 1][k] = 0;
            }
        }
        for (int j = 0; j < res[1]; j++) {
            for (int k = 0; k < res[2]; k++) {
                u[0][j][k] = u[res[0] - 1][j][k] = 0;
            }
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

TC_IMPLEMENTATION(Simulation3D, Smoke3D, "smoke");

TC_NAMESPACE_END
