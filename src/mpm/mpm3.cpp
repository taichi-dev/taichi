#include "mpm3.h"
#include "math/qr_svd/qr_svd.h"
#include "system/threading.h"

TC_NAMESPACE_BEGIN

// Note: assuming abs(x) <= 2!!
inline float w(float x) {
	x = abs(x);
	assert(x <= 2);
	if (x < 1) {
		return 0.5f * x * x * x - x * x + 2.0f / 3.0f;
	}
	else {
		return -1.0f / 6.0f * x * x * x + x * x - 2 * x + 4.0f / 3.0f;
	}
}

// Note: assuming abs(x) <= 2!!
inline float dw(float x) {
	float s = x < 0.0f ? -1.0f : 1.0f;
	x *= s;
	assert(x <= 2.0f);
	float val;
	float xx = x * x;
	if (x < 1.0f) {
		val = 1.5f * xx - 2.0f * x;
	}
	else {
		val = -0.5f * xx + 2.0f * x - 2.0f;
	}
	return s * val;
}

inline float w(const Vector3 &a) {
	return w(a.x) * w(a.y) * w(a.z);
}

inline Vector3 dw(const Vector3 &a) {
	return Vector3(dw(a.x) * w(a.y) * w(a.z), w(a.x) * dw(a.y) * w(a.z), w(a.x) * w(a.y) * dw(a.z));
}

long long MPM3D::Particle::instance_count;

struct EPParticle3 : public MPM3D::Particle {
	EPParticle3() : MPM3D::Particle() {
	}
	virtual Matrix get_energy_gradient() {
		const real hardening = 10.0f;
		const real mu_0 = 1e5f, lambda_0 = 1e5f;
		real j_e = det(dg_e);
		real j_p = det(dg_p);
		real e = std::exp(std::min(hardening * (1.0f - j_p), 5.0f));
		real mu = mu_0 * e;
		real lambda = lambda_0 * e;
		Matrix r, s;
		polar_decomp(dg_e, r, s);
		if (!is_normal(r)) {
			P(dg_e);
			P(r);
			P(s);
		}
		CV(r);
		CV(s);
		return 2 * mu * (dg_e - r) +
			lambda * (j_e - 1) * j_e * glm::inverse(glm::transpose(dg_e));
	}
	virtual void calculate_kernels() {}
	virtual void calculate_force() {
		tmp_force = -vol * get_energy_gradient() * glm::transpose(dg_e);
	};
	virtual void plasticity() {
		Matrix svd_u, sig, svd_v;
		svd(dg_e, svd_u, sig, svd_v);
		const float theta_c = 2.5e-2f, theta_s = 7.5e-3f;
		for (int i = 0; i < D; i++) {
			sig[i][i] = clamp(sig[i][i], 1.0f - theta_c, 1.0f + theta_s);
		}
		dg_e = svd_u * sig * glm::transpose(svd_v);
		dg_p = glm::inverse(dg_e) * dg_cache;
		svd(dg_p, svd_u, sig, svd_v);
		for (int i = 0; i < D; i++) {
			sig[i][i] = clamp(sig[i][i], 0.01f, 10.0f);
		}
		dg_p = svd_u * sig * glm::transpose(svd_v);
	};
};

void MPM3D::initialize(const Config &config) {
	Simulation3D::initialize(config);
	res = config.get_vec3i("resolution");
	P(res);
	gravity = config.get_vec3("gravity");
	delta_t = config.get_real("delta_t");

	t = 0.0f;

	auto initial_velocity = config.get_vec3("initial_velocity");
	for (int i = int(res[0] * 0.4); i <= int(res[0] * 0.6); i++) {
		for (int j = int(res[1] * 0.2); j <= int(res[1] * 0.6); j++) {
			for (int k = int(res[2] * 0.4); k <= int(res[2] * 0.6); k++) {
				if (j < res[1] / 2 && i < res[0] / 2) {
					continue;
				}
				for (int l = 0; l < 8; l++) {
					Particle *p = new EPParticle3();
					p->pos = Vector(i + rand(), j + rand(), k + rand());
					p->mass = 1.0f;
					p->v = initial_velocity;
					particles.push_back(p);
				}
			}
		}
	}
	grid_velocity.initialize(res[0], res[1], res[2], Vector(0.0f));
	grid_mass.initialize(res[0], res[1], res[2], 0);
	grid_locks.initialize(res[0], res[1], res[2], 0);

	/*
	for (int i = 0; i < 100; i++) {
		Matrix3 m(0.0f), u, s, v, r;
		for (int j = 0; j < 9; j++) {
			m[j / 3][j % 3] = rand();
		}
		svd(m, u, s, v);
		r = m - u * s * transpose(v);
		P(m);
		P(u);
		P(s);
		P(v);
		P(r);
		P(frobenius_norm(r));
	}
	*/
}

std::vector<RenderParticle> MPM3D::get_render_particles() const {
	using Particle = RenderParticle;
	std::vector<Particle> render_particles;
	render_particles.reserve(particles.size());
	Vector3 center(res[0] / 2.0f, res[1] / 2.0f, res[2] / 2.0f);
	for (auto p_p : particles) {
		MPM3D::Particle &p = *p_p;
		render_particles.push_back(Particle(p.pos - center, Vector4(0.8f, 0.9f, 1.0f, 0.5f)));
	}
	return render_particles;
}

void MPM3D::rasterize() {
	grid_velocity.reset(Vector(0.0f));
	grid_mass.reset(0.0f);
	parallel_for_each_particle([&](Particle &p) {
		for (auto &ind : get_bounded_rasterization_region(p.pos)) {
			Vector3 d_pos = Vector(ind.i, ind.j, ind.k) - p.pos;
			real weight = w(d_pos);
			grid_locks[ind].lock();
			grid_mass[ind] += weight * p.mass;
			grid_velocity[ind] += weight * p.mass * (p.v + (3.0f) * p.apic_b * d_pos);
			grid_locks[ind].unlock();
		}
	});
	for (auto ind : grid_mass.get_region()) {
		if (grid_mass[ind] > 0) {
			CV(grid_velocity[ind]);
			CV(1 / grid_mass[ind]);
			grid_velocity[ind] = grid_velocity[ind] * (1.0f / grid_mass[ind]);
			CV(grid_velocity[ind]);
		}
	}
}

void MPM3D::resample(float delta_t) {
	parallel_for_each_particle([&](Particle &p) {
		Vector v(0.0f);
		Matrix cdg(0.0f);
		Matrix b(0.0f);
		for (auto &ind : get_bounded_rasterization_region(p.pos)) {
			Vector d_pos = p.pos - Vector3(ind.i, ind.j, ind.k);
			float weight = w(d_pos);
			Vector gw = dw(d_pos);
			v += weight * grid_velocity[ind];
			Vector aa = grid_velocity[ind];
			Vector bb = -d_pos;
			Matrix out(aa[0] * bb[0], aa[1] * bb[0], aa[2] * bb[0],
				aa[0] * bb[1], aa[1] * bb[1], aa[2] * bb[1],
				aa[0] * bb[2], aa[1] * bb[2], aa[2] * bb[2]);
			b += weight * out;
			cdg += glm::outerProduct(grid_velocity[ind], gw);
			CV(grid_velocity[ind]);
		}
		p.apic_b = b;
		cdg = Matrix(1) + delta_t * cdg;
		p.v = v;
		Matrix dg = cdg * p.dg_e * p.dg_p;
		p.dg_e = cdg * p.dg_e;
		p.dg_cache = dg;
	});
}

void MPM3D::apply_deformation_force(float delta_t) {
	//printf("Calculating force...\n");
	parallel_for_each_particle([&](Particle &p) {
		p.calculate_force();
	});
	//printf("Accumulating force...\n");
	parallel_for_each_particle([&](Particle &p) {
		for (auto &ind : get_bounded_rasterization_region(p.pos)) {
			real mass = grid_mass[ind];
			if (mass == 0.0f) { // No EPS here
				continue;
			}
			Vector d_pos = p.pos - Vector3(ind.i, ind.j, ind.k);
			Vector gw = dw(d_pos);
			Vector force = p.tmp_force * gw;
			CV(force);
			grid_locks[ind].lock();
			grid_velocity[ind] += delta_t / mass * force;
			grid_locks[ind].unlock();
		}
	});
}

void MPM3D::apply_boundary_conditions() {

}

void MPM3D::substep(float delta_t) {
	if (!particles.empty()) {
		/*
		for (auto &p : particles) {
			p.calculate_kernels();
		}
		*/
		apply_external_impulse(gravity * delta_t);
		rasterize();
		apply_deformation_force(delta_t);
		apply_boundary_conditions();
		resample(delta_t);
		parallel_for_each_particle([&](Particle &p) {
			p.pos += delta_t * p.v;
			p.pos.x = clamp(p.pos.x, 0.0f, res[0] - eps);
			p.pos.y = clamp(p.pos.y, 0.0f, res[1] - eps);
			p.pos.z = clamp(p.pos.z, 0.0f, res[2] - eps);
			p.plasticity();
		});
	}
	current_t += delta_t;
}

TC_IMPLEMENTATION(Simulation3D, MPM3D, "mpm");

TC_NAMESPACE_END
