#pragma once
#include <memory.h>
#include <string>
#include "visualization/image_buffer.h"
#include "common/config.h"
#include "common/interface.h"

TC_NAMESPACE_BEGIN

class Fluid : public Simulator {
public:
	virtual void show(ImageBuffer<Vector3> &buffer) = 0;
	struct Particle {
		Vector3 color=Vector3(-1, 0, 0);
		Vector2 position, velocity;
		Vector2 weight;
		Vector2 c[2] = { Vector2(0), Vector2(0) }; // for APIC
		long long id = instance_counter++;
		bool show = true;
		float temperature;
		float radius = 0.75f;
		Particle() {};
		Particle(Vector2 position, Vector2 velocity = Vector2(0)) : position(position), velocity(velocity) { }
		void move(Vector2 delta_x) {
			this->position += delta_x;
		}

		template <int k> static float get_velocity(const Particle &p, const Vector2 &delta_pos) { return p.velocity[k]; }
		template <int k> static float get_affine_velocity(const Particle &p, const Vector2 &delta_pos) { return p.velocity[k] + dot(p.c[k], delta_pos); }
		static float get_signed_distance(const Particle &p, const Vector2 &delta_pos) { return length(delta_pos) - p.radius; }

		static long long instance_counter;

		bool operator == (const Particle &o) {
			return o.id == id;
		}
	};
protected:
    std::vector<Particle> particles;
};

std::shared_ptr<Fluid> create_fluid(std::string name);

TC_NAMESPACE_END

