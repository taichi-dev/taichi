#include "mpm.h"

TC_NAMESPACE_BEGIN

template <void(*T)(const mat2 &, mat2 &, mat2&, mat2&)>
void svd_test() {
	int test_num = 100000000;
	int error_count = 0;
	for (int k = 0; k < test_num; k++) {
		mat2 m;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				m[i][j] = rand() * 2 - 1;
			}
		}
		mat2 u, sig, v;
		T(m, u, sig, v);
		if (frobenius_norm(m - u * sig * glm::transpose(v)) > 1e-4f) {
			if (error_count < 10) {
				P(m);
				P(u);
				P(sig);
				P(v);
			}
			error_count++;
		}
	}
	printf("SVD Test error: %d / %d\n", error_count, test_num);
}

void MPM::initialize(const Config &config_) {
	auto config = Config(config_);
	this->config = config;
	width = config.get_int("simulation_width");
	height = config.get_int("simulation_height");
	this->implicit_ratio = config.get_float("implicit_ratio");
	this->apic = config.get("apic", true);
	this->use_level_set = config.get("use_level_set", false);
	this->cfl = config.get("cfl", 0.01f);
	this->h = config.get_float("delta_x");
	grid.initialize(width, height, implicit_ratio != 0);
	t = 0.0f;
	last_sort = 1e20f;
	flip_alpha = config.get_float("flip_alpha");
	flip_alpha_stride = config.get_float("flip_alpha_stride");
	gravity = config.get_vec2("gravity");
	max_delta_t = config.get("max_delta_t", 0.001f);
	min_delta_t = config.get("min_delta_t", 0.00001f);
	material_levelset.initialize(width, height, Vector2(0.5f, 0.5f));
}

void MPM::substep(float delta_t) {
	if (!particles.empty()) {
		for (auto &p : particles) {
			p->calculate_kernels();
		}
		rasterize();
		grid.reorder_grids();
		estimate_volume();
		grid.backup_velocity();
		grid.apply_external_force(gravity, delta_t);
		//Deleted: grid.apply_boundary_conditions(levelset);
		apply_deformation_force(delta_t);
		if (implicit_ratio > 0.0f)
			implicit_velocity_update(delta_t);
		grid.apply_boundary_conditions(levelset);
		resample(delta_t);
		for (auto &p : particles) {
			p->pos += delta_t * p->v;
		}
		if (config.get("particle_collision", false))
			particle_collision_resolution();
	}
	t += delta_t;
}

void MPM::step(float delta_t)
{
	float simulation_time = 0.0f;
	while (simulation_time < delta_t - eps) {
		float purpose_dt = std::min(max_delta_t, get_dt_with_cfl_1() * cfl);
		float thres = min_delta_t;
		if (purpose_dt < delta_t * thres) {
			purpose_dt = delta_t * thres;
			printf("substep dt too small, clamp.\n");
		}
		float dt = std::min(delta_t - simulation_time, purpose_dt);
		substep(dt);
		simulation_time += dt;
	}
	compute_material_levelset();
}

void MPM::compute_material_levelset()
{
	material_levelset.reset(std::numeric_limits<float>::infinity());
	for (auto &p : particles) {
		for (auto &ind : material_levelset.get_rasterization_region(p->pos, 3)) {
			Vector2 delta_pos = ind.get_pos() - p->pos;
			material_levelset[ind] = std::min(material_levelset[ind], length(delta_pos) - 0.8f);
		}
	}
	for (auto &ind : material_levelset.get_region()) {
		if (material_levelset[ind] < 0.5f) {
			if (levelset.sample(ind.get_pos()) < 0)
				material_levelset[ind] = -0.5f;
		}
	}
}

void MPM::particle_collision_resolution() {
	for (auto &p : particles) {
		p->resolve_collision(levelset);
	}
}

void MPM::estimate_volume() {
	for (auto &p : particles) {
		if (p->vol == -1.0f) {
			float rho = 0.0f;
			for (auto &ind : get_bounded_rasterization_region(p->pos)) {
				float weight = p->get_cache_w(ind);
				rho += grid.mass[ind] / h / h;
			}
			p->vol = p->mass / rho;
		}
	}
}

void MPM::show(ImageBuffer<Vector3>& buffer) {
}

void MPM::add_particle(const Config &config)
{
	auto p = create_particle(config);
	p->mass = 1.0f / width / width;
	// p->pos += config.get("position_noise", 0.0f) * Vector2(rand() - 0.5f, rand() - 0.5f);
	particles.push_back(p);
}

void MPM::add_particle(std::shared_ptr<MPMParticle> p)
{
	p->mass = 1.0f / width / width;
	p->pos += config.get("position_noise", 0.0f) * Vector2(rand() - 0.5f, rand() - 0.5f);
	particles.push_back(p);
}

void MPM::add_particle(EPParticle p) {
	add_particle(std::make_shared<EPParticle>(p));
}

void MPM::add_particle(DPParticle p) {
	add_particle(std::make_shared<DPParticle>(p));
}

std::vector<std::shared_ptr<MPMParticle>> MPM::get_particles()
{
	return particles;
}

float MPM::get_current_time() {
	return t;
}

LevelSet2D MPM::get_material_levelset()
{
	return material_levelset;
}

void MPM::rasterize() {
	grid.reset();
	for (auto &p : particles) {
		if (!is_normal(p->pos)) {
			p->print();
		}
		//if (t > 3.15f)
		//	P(p->pos);
		for (auto &ind : get_bounded_rasterization_region(p->pos)) {
			//if (t > 3.15f)
			//	printf("in %d %d\n", ind.i, ind.j);
			float weight = p->get_cache_w(ind);
			grid.mass[ind] += weight * p->mass;
			grid.velocity[ind] += weight * p->mass * (p->v + (3.0f) * p->b * (Vector2(ind.i, ind.j) - p->pos));
		}
	}
	grid.normalize_velocity();
}

inline void MPM::resample(float delta_t) {
	float alpha_delta_t = pow(flip_alpha, delta_t / flip_alpha_stride);
	if (apic)
		alpha_delta_t = 0.0f;
	for (auto &p : particles) {
		int p_i = int(p->pos.x);
		int p_j = int(p->pos.y);
		vec2 v = vec2(0, 0), bv = vec2(0, 0);
		mat2 cdg(0.0f);
		mat2 b(0.0f);
		int count = 0;
		for (auto &ind : get_bounded_rasterization_region(p->pos)) {
			count++;
			float weight = p->get_cache_w(ind);
			vec2 gw = p->get_cache_gw(ind);
			v += weight * grid.velocity[ind];
			vec2 aa = grid.velocity[ind];
			vec2 bb = Vector2(ind.i, ind.j) - p->pos;
			mat2 out(aa[0] * bb[0], aa[1] * bb[0], aa[0] * bb[1], aa[1] * bb[1]);
			b += weight * out;
			bv += weight * grid.velocity_backup[ind];
			cdg += glm::outerProduct(grid.velocity[ind], gw);
		}
		if (count != 16 || !apic) {
			b = mat2(0.0f);
		}
		CV(cdg);
		p->b = b;
		cdg = mat2(1.0f) + delta_t * cdg;

		p->v = (1 - alpha_delta_t) * v + alpha_delta_t * (v - bv + p->v);
		mat2 dg = cdg * p->dg_e * p->dg_p;
		p->dg_e = cdg * p->dg_e;
		p->dg_cache = dg;
		//if (!is_normal(p.v)) {
		//	printf("Velocity Abnormal!\n");
		//	Pp(cdg);
		//	for (int i = p_i - 1; i < p_i + 3; i++) {
		//		for (int j = p_j - 1; j < p_j + 3; j++) {
		//			if (0 <= i && i < dim && 0 <= j && j < dim) {
		//				Pp(i);
		//				Pp(j);
		//				float weight = p.cache_w[CACHE_INDEX];
		//				vec2 gw = p.cache_gw[CACHE_INDEX];
		//				Pp(weight * grid.velocity[i][j]);
		//				Pp(grid.velocity[i][j]);
		//				Pp(weight * grid.velocity_backup[i][j]);

		//				Pp(weight);
		//				Pp(gw);
		//			}
		//		}
		//	}
		//	assert(false);
		//}
	}
	for (auto &p : particles) {
		p->plasticity();
	}
}

mat4 MPM::get_energy_second_derivative_brute_force(Particle & p, float delta) {
	NOT_IMPLEMENTED;
	return mat4(0.0f);
	//	mat4 ret(0);
	//for (int i = 0; i < 2; i++) {
	//	for (int j = 0; j < 2; j++) {
	//		p.dg_e[i][j] += delta;
	//		mat2 gradient_1 = get_energy_gradient(p);
	//		p.dg_e[i][j] -= 2 * delta;
	//		mat2 gradient_0 = get_energy_gradient(p);
	//		mat2 dd = (gradient_1 - gradient_0) / (delta * 2);
	//		for (int k = 0; k < 4; k++) {
	//			ret[k][j * 2 + i] = dd[k % 2][k / 2];
	//		}
	//		p.dg_e[i][j] += delta;
	//	}
	//}
	//return ret;
}

mat4 MPM::get_energy_second_derivative(Particle & p) {
	NOT_IMPLEMENTED;
	return mat4(0.0f);
	// This code is optimized and without readibility. Plz refer to ealier version.
	/*
	const mat2 &f = p.dg_e;
	const float j_e = det(p.dg_e);
	const float j_p = det(p.dg_p);
	const float e = expf(hardening * (1.0f - j_p));
	const float mu = mu_0 * e;
	const float lambda = lambda_0 * e;
	mat2 r, s;
	polar_decomp(p.dg_e, r, s);
	mat4 sum(0);
	sum = mat4(2 * mu);
	vec4 x = vec4(-r[1][0], r[0][0], -r[1][1], r[0][1]);
	mat4 m_dR = glm::outerProduct(2.0f / (s[0][0] + s[1][1]) * mu * vec4(r[1][0], -r[0][0], r[1][1], -r[0][1]),
		vec4(x));
	sum += m_dR;
	sum += glm::outerProduct(lambda * vec4(f[1][1], -f[0][1], -f[1][0], f[0][0]),
		vec4(f[1][1], -f[0][1], -f[1][0], f[0][0]));
	float t = lambda * (j_e - 1);
	sum[0][3] += t;
	sum[1][2] -= t;
	sum[2][1] -= t;
	sum[3][0] += t;
	return sum;
	*/
}

void MPM::build_system(const float delta_t) {
	NOT_IMPLEMENTED;
	//	system.reset(grid.valid_count);
	//	Time::TickTimer _("build system");
	//	if ((unsigned long long)(void *)&grid.system(0, 0, 0, 0) % 16 != 0) {
	//		assert("false");
	//	}
	//
	//	// NOTE: Potential racing condition errors!
	//#pragma omp parallel for
	//	for (int t = 0; t < particles.size(); t++) {
	//		auto &p = particles[t];
	//		const mat4 cache_phi = get_energy_second_derivative(p) * p.vol;
	//		// Pp(cache_phi - glm::transpose(cache_phi));
	//		const int p_i = int(p->pos.x);
	//		const int p_j = int(p->pos.y);
	//		for (auto &ind : get_bounded_rasterization_region(p.pos)) {
	//			int i = ind.i, j = ind.j;
	//			int id_0 = grid.id[ind];
	//			if (id_0 == -1)
	//				continue;
	//			const float factor = implicit_ratio * delta_t * delta_t;
	//			const vec2 Ftdwj = p->get_cache_gw(ind);
	//			__declspec(align (16)) float accumulator[2][2][2] = { 0 };
	//			/*
	//			for (int tao = 0; tao < 2; tao++) {
	//			for (int sigma = 0; sigma < 2; sigma++) {
	//			for (int beta = 0; beta < 2; beta++) {
	//			accumulator[tao][sigma][0] += cache_phi[sigma * 2 + beta][tao * 2 + 0] * Ftdwj[beta];
	//			accumulator[tao][sigma][1] += cache_phi[sigma * 2 + beta][tao * 2 + 1] * Ftdwj[beta];
	//			}
	//			}
	//			}
	//			*/
	//#define CALC_ACCUMULATOR(tao, sigma, kai) accumulator[kai][tao][sigma] = factor * (\
	// cache_phi[sigma * 2 + 0][tao * 2 + kai] * Ftdwj[0] +\
	// cache_phi[sigma * 2 + 1][tao * 2 + kai] * Ftdwj[1]);
	//			CALC_ACCUMULATOR(0, 0, 0);
	//			CALC_ACCUMULATOR(0, 0, 1);
	//			CALC_ACCUMULATOR(0, 1, 0);
	//			CALC_ACCUMULATOR(0, 1, 1);
	//			CALC_ACCUMULATOR(1, 0, 0);
	//			CALC_ACCUMULATOR(1, 0, 1);
	//			CALC_ACCUMULATOR(1, 1, 0);
	//			CALC_ACCUMULATOR(1, 1, 1);
	//			__m128 g_x, g_y, mat_x, mat_y;
	//			mat_x = _mm_load_ps((float *)accumulator[0]);
	//			mat_y = _mm_load_ps((float *)accumulator[1]);
	//			for (int k = p_i - 1; k < p_i + 3; k++) {
	//				float *target = (float *)&grid.system(i, j, k, p_j - 1);
	//				int *p_id = &grid.id[k][p_j - 1];
	//				for (int l = p_j - 1; l < p_j + 3; l++) {
	//					if (0 <= k && k < dim && 0 <= l && l < dim) {
	//						int id_1 = *p_id;
	//						if (id_0 <= id_1) {
	//							__m128 entry = _mm_load_ps(target);
	//							const vec2 Ftdwk = p->cache_dg_gw[(k - p_i + 1) * 4 + (l - p_j + 1)];
	//							g_x = _mm_set1_ps(Ftdwk[0]);
	//							g_y = _mm_set1_ps(Ftdwk[1]);
	//							__m128 ret_x = _mm_mul_ps(mat_x, g_x);
	//							__m128 ret = _mm_add_ps(ret_x, _mm_mul_ps(mat_y, g_y));
	//							_mm_store_ps(target, _mm_add_ps(entry, ret));
	//						}
	//					}
	//					target += 4;
	//					p_id++;
	//				}
	//			}
	//		}
	//	}
	//	system.rhs = ArrayVec2(grid.valid_count);
	//	for (int i = 0; i < dim; i++) {
	//		for (int j = 0; j < dim; j++) {
	//			int id_0 = grid.id[i][j];
	//			if (id_0 == -1) {
	//				continue;
	//			}
	//			float inv_mass_0 = 1.0f / grid.mass[i][j];
	//			system.rhs[id_0] = grid.velocity[i][j];
	//			for (int k = i - 3; k <= i + 3; k++) {
	//				for (int l = j - 3; l <= j + 3; l++) {
	//					if (0 <= k && k < dim && 0 <= l && l < dim) {
	//						int id_1 = grid.id[k][l];
	//						if (id_1 == -1 && id_1 < id_0)
	//							continue;
	//						float inv_mass_1 = 1.0f / grid.mass[k][l];
	//						const mat2 &val = grid.system(i, j, k, l);
	//						if (val != mat2(0)) {
	//							if (i == k && j == l) {
	//								system.append(id_0, id_1, val * inv_mass_0 + mat2(1));
	//							}
	//							else {
	//								system.append(id_0, id_1, val * inv_mass_0);
	//							}
	//							if (id_1 != id_0) {
	//								system.append(id_1, id_0, glm::transpose(val) * inv_mass_1);
	//							}
	//						}
	//					}
	//				}
	//			}
	//		}
	//	}
}

void MPM::apply_A(const ArrayVec2 & x, ArrayVec2 & p) {
	p = system.apply(x);
}

// CR solver

ArrayVec2 MPM::solve_system(ArrayVec2 x_0, Grid & grid) { // returns: total error
	system.precondition();
	int size = grid.valid_count;
	ArrayVec2 x(size), r(size), Ax(size), p(size), Ar(size), Ap(size);
	vector<float> mass(size);
	for (int i = 0; i < size; i++) {
		ivec2 g = grid.id_to_pos[i];
		mass[i] = (grid.mass[g.x][g.y]);
	}
	x = x_0;
	apply_A(x, Ax);
	r = system.rhs - Ax;
	p = r;
	apply_A(r, Ar);
	Ap = Ar;
	float rtAr = r.dot(Ar);
	bool early_break = false;
	for (int k = 0; k < config.get_int("maximum_iterations"); k++) {
		float Ap_sqr = Ap.dot(Ap) + 1e-10f;
		float alpha = rtAr / Ap_sqr;
		x = x.add(alpha, p);
		r = r.add(-alpha, Ap);
		float error = 0.0f;
		for (int i = 0; i < size; i++) {
			error += mass[i] * glm::length(p[i]);
		}
		if (abs(error * alpha) < 1e-8f) {
			printf("CR converged at iteration %d\n", k + 1);
			early_break = true;
			break;
		}
		if (k > 100 && false) {
			Pp(error);
			int max_index = -1;
			for (int i = 0; i < size; i++) {
				if (max_index == -1 || glm::length(p[i]) > glm::length(p[max_index])) {
					max_index = i;
				}
			}
			// Pp(max_index);
			ivec2 g = grid.id_to_pos[max_index];
			Pp(grid.mass[g.x][g.y]);
		}

		apply_A(r, Ar);
		float new_rtAr = r.dot(Ar);
		float beta = new_rtAr / rtAr;
		rtAr = new_rtAr;
		p = r.add(beta, p);
		Ap = Ar.add(beta, Ap);
	}
	if (!early_break) {
		printf("Warning: CR iteration exceeds upper limit\n");
	}
	return x;
}

void MPM::implicit_velocity_update(const float & delta_t) {
	build_system(delta_t);
	ArrayVec2 initial(grid.valid_count);
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			int id = grid.id[i][j];
			if (id == -1)
				continue;
			initial[id] = grid.velocity[i][j];
		}
	}
	ArrayVec2 sol = solve_system(initial, grid);
	for (int id = 0; id < grid.valid_count; id++) {
		ivec2 pos = grid.id_to_pos[id];
		grid.velocity[pos.x][pos.y] = sol[id];
	}
}

void MPM::apply_deformation_force(float delta_t) {
#pragma omp parallel for
	for (auto &p : particles) {
		p->calculate_force();
	}
	// NOTE: Potential racing condition errors!
#pragma omp parallel for
	for (auto &p : particles) {
		for (auto &ind : get_bounded_rasterization_region(p->pos)) {
			float mass = grid.mass[ind];
			if (mass == 0.0f) { // No EPS here
				continue;
			}
			vec2 gw = p->get_cache_gw(ind);
			vec2 force = p->tmp_force * gw;
			grid.velocity[ind] += delta_t / mass * force;
		}
	}
}

float MPM::get_dt_with_cfl_1()
{
	return 1 / max(get_max_speed(), 1e-5f);
}

float MPM::get_max_speed()
{
	float maximum_speed = 0;
	for (auto &p : particles) {
		maximum_speed = max(abs(p->v.x), maximum_speed);
		maximum_speed = max(abs(p->v.y), maximum_speed);
	}
	return maximum_speed;
}

TC_NAMESPACE_END

