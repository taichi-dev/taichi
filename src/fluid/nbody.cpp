#pragma once

#include "simulation3d.h"
#include "particle_visualization.h"
#include "renderer/texture.h"

TC_NAMESPACE_BEGIN

class BarnesHutSummation {
protected:
	struct Particle {
		Vector3 position;
		real mass;
		Particle() { mass = 0.0f; position = Vector3(0.0f); }
		Particle(const Vector3 &position, real &mass) : position(position), mass(mass) {}
		Particle operator + (const Particle &o) {
			Particle ret;
			ret.mass = mass + o.mass;
			ret.position = (position * mass + o.position * o.mass) / ret.mass;
			return ret;
		}
	};
	struct Node {
		Particle p;
		int children[8]; // We do not use pointer here to save memory (64 bit v.s. 32 bit)
		Node() {
			memset(children, 0, sizeof(children));
		}
	};
	real resolution, inv_resolution;
	int total_level;

	std::vector<Node> nodes;
	int node_end;
	Vector3 lower_corner;
public:

	Vector3i get_coord(const Vector3 &position) {
		Vector3i u;
		for (int i = 0; i < 3; i++) {
			u[i] = int((position[i] - lower_corner[i]) * inv_resolution);
		}
		return u;
	}

	int get_child_index(const Vector3i &u, int level) {
		int ret = 0;
		for (int i = 0; i < 3; i++) {
			ret += ((u[i] >> (total_level - level + 1)) & 1) << i;
		}
		return ret;
	}

	// We do not evaluate the weighted average of position and mass
	// for efficiency and accuracy
	void initialize(real resolution, const std::vector<Particle> &particles) {
		this->resolution = resolution;
		inv_resolution = 1.0f / resolution;
		assert(particles.size() != 0);
		Vector3 lower(1e30f);
		Vector3 upper(-1e30f);
		for (auto &p : particles) {
			for (int k = 0; k < 3; k++) {
				lower[k] = std::min(lower[k], p.position[k]);
				upper[k] = std::max(upper[k], p.position[k]);
			}
		}
		lower_corner = lower;
		int tmp = (int)std::ceil(max_component(upper - lower) / resolution);
		total_level = 0;
		for (int i = 1; i <= tmp; i *= 2, total_level++);
		node_end = 0;
		nodes.resize(particles.size() * 2);
		int root = get_new_node();
		// Make sure that one leaf node contains only one particle.
		// Unless particles are too close thereby merged.
		// TODO:....
		for (auto &p : particles) {
			if (p.mass == 0) {
				continue;
			}
			Vector3i u = get_coord(p.position);
			int t = root;
			Node node = nodes[t];
			for (int k = 0; k < total_level; k++) {
				int c = get_child_index(u, k);
				if (node.children[c] == 0) {
					node.children[c] = get_new_node();
				}
				t = node.children[c];
			}
		}
	}

	template<typename T>
	Vector3 summation(const Particle &p) {
		// TODO: fine level	
		int t = root;
		for (int k = 0; k < total_level; k++) {
			for (int c = 0; c < 8; c++) {
				if (node.children[c])
			}
		}
	}

	int get_new_node() {
		return node_end++;
	}
};

class NBody : public Simulation3D {
	struct Particle {
		Vector3 position, velocity, color;
		Particle(const Vector3 &position, const Vector3 &velocity, const Vector3 &color) :
			position(position), velocity(velocity), color(color) {
		}
	};
protected:
	real gravitation;
	std::shared_ptr<Texture> velocity_field;
	std::vector<Particle> particles;
	real delta_t;
public:
	virtual void initialize(const Config &config) override {
		int num_particles = config.get_int("num_particles");
		particles.reserve(num_particles);
		gravitation = config.get_real("gravitation");
		delta_t = config.get_real("delta_t");
		real vel_scale = config.get_real("vel_scale");
		for (int i = 0; i < num_particles; i++) {
			Vector3 p(rand(), rand(), rand());
			Vector3 v = Vector3(p.y, p.z, p.x) - Vector3(0.5f);
			v *= vel_scale;
			Vector3 c(0.5, 0.7, 0.4);
			particles.push_back(Particle(p, v, c));
		}
	}
	std::vector<RenderParticle> get_render_particles() const override {
		std::vector<RenderParticle> render_particles;
		render_particles.reserve(particles.size());
		for (auto &p : particles) {
			render_particles.push_back(RenderParticle(p.position - Vector3(0.5f), p.color));
		}
		return render_particles;
	}
	void substep(real dt) {
		for (auto &p : particles) {
			p.position += dt * p.velocity;
		}
		if (gravitation != 0) {
			for (int i = 0; i < (int)particles.size(); i++) {
				auto &p = particles[i];
				for (int j = i + 1; j < (int)particles.size(); j++) {
					auto &q = particles[j];
					Vector3 d = p.position - q.position;
					real dist2 = dot(d, d);
					dist2 += 1e-10f;
					d *= gravitation * dt / (dist2 * sqrt(dist2));
					p.velocity += d;
					q.velocity -= d;
				}
			}
		}
		current_t += dt;
	}
	virtual void step(real dt) override {
		int steps = (int)std::ceil(dt / delta_t);
		for (int i = 0; i < steps; i++) {
			substep(dt / steps);
		}
	}
};

TC_IMPLEMENTATION(Simulation3D, NBody, "nbody");

TC_NAMESPACE_END
