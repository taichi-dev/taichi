#pragma once

#include <memory>
#include <vector>
#include "mpm_grid.h"
#include <taichi/levelset/levelset2d.h>
#include <taichi/visualization/image_buffer.h>

TC_NAMESPACE_BEGIN

#define CACHE_INDEX ((i - p_i + 1) * 4 + (j - p_j + 1))

struct MPMLinearSystemRow {
	int num_items;
	mat2 items[49];
	int indices[49];
	void reset() {
		num_items = 0;
	}
	void append(int index, const mat2 &item) {
		/*
		int i = 0;
		if (num_items > 0)
			for (i = num_items; i >= 1 && indices[i - 1] > index; i--) {
				indices[i] = indices[i - 1];
				items[i] = items[i - 1];
			}
		indices[i] = index;
		items[i] = item;
		num_items++;
		*/
		indices[num_items] = index;
		items[num_items] = item;
		num_items++;
	}
};

class MPMLinearSystem {
public:
	std::vector<MPMLinearSystemRow> data;
	ArrayVec2 rhs;
	std::vector<mat2> diag;
	int size;
	void reset(int size) {
		this->size = size;
		data.resize(size);
		diag.resize(size);
		for (int i = 0; i < size; i++) {
			data[i].reset();
		}
	}
	ArrayVec2 apply(const ArrayVec2 &x) {
		// Time::TickTimer _("apply system");
		ArrayVec2 y(size, vec2(0));
		for (int i = 0; i < size; i++) {
			vec2 tmp_0(0), tmp_1(0);
			const int &num_items = data[i].num_items;
			for (int j = 0; j < num_items; j += 1) {
				tmp_0 += data[i].items[j] * x[data[i].indices[j]];
			}
			y[i] = tmp_0 + tmp_1;
			CV(y[i]);
		}
		return y;
	}
	void append(int row, int column, const mat2 &item) {
		data[row].append(column, item);
		if (row == column) {
			diag[row] = item;
		}
	}
	void precondition() {
		for (int i = 0; i < size; i++) {
			mat2 precond = glm::inverse(diag[i]);
			const int &num_items = data[i].num_items;
			for (int j = 0; j < num_items; j += 1) {
				data[i].items[j] *= precond;
			}
			rhs[i] = precond * rhs[i];
		}
	}

};

class MPM {
protected:
	Config config;
	float theta_c, theta_s;
	std::vector<std::shared_ptr<Particle>> particles;
	Grid grid;
	int dim;
	int width;
	int height;
	float flip_alpha;
	float flip_alpha_stride;
	float h;
	float t;
	float last_sort;
	float sorting_period;
	vec2 gravity;
	float implicit_ratio;
	MPMLinearSystem system;
	bool apic;
	bool use_level_set;
	float max_delta_t;
	float min_delta_t;
	LevelSet2D levelset;
	LevelSet2D material_levelset;

	void compute_material_levelset();

	Region2D get_bounded_rasterization_region(Vector2 p) {
		int x = int(p.x);
		int y = int(p.y);
		int x_min = std::max(0, x - 1);
		int x_max = std::min(width, x + 3);
		int y_min = std::max(0, y - 1);
		int y_max = std::min(height, y + 3);
		return Region2D(x_min, x_max, y_min, y_max);
	}

	void particle_collision_resolution();

	void estimate_volume();

	void rasterize();

	void resample(float delta_t);

	mat4 get_energy_second_derivative_brute_force(Particle &p, float delta = 1e-2f);

	mat4 get_energy_second_derivative(Particle &p);

	void build_system(const float delta_t);

	void apply_A(const ArrayVec2 &x, ArrayVec2 &p);

	// CR solver
	ArrayVec2 solve_system(ArrayVec2 x_0, Grid &grid);

	void implicit_velocity_update(const float &delta_t);

	void apply_deformation_force(float delta_t);

	virtual void substep(float delta_t);

	float get_dt_with_cfl_1();

	float get_max_speed();

	float cfl;

public:
	MPM() {
		sorting_period = 1.0f;
	}

	void initialize(const Config &config_);

	void step(float delta_t = 0.0f);

	void show(ImageBuffer<Vector3> &buffer);

	void add_particle(const Config &config);

	void add_particle(std::shared_ptr<MPMParticle> particle);

	void add_particle(EPParticle p);

	void add_particle(DPParticle p);

	std::vector<std::shared_ptr<MPMParticle>> get_particles();

	float get_current_time();

	void set_levelset(const LevelSet2D &levelset) {
		this->levelset = levelset;
	}

	LevelSet2D get_material_levelset();

};

TC_NAMESPACE_END

