#pragma once
#include "flip_fluid.h"
#include "voronoi_flip_fluid.h"
#include "point_cloud/point_cloud.h"

TC_NAMESPACE_BEGIN

class FLIPSmoke : public FLIPFluid {
protected:
	float ambient_temp;
	float buoyancy;
	float conduction;
	bool show_air_particles;
	Array temperature;
	Array temperature_backup;
	Array temperature_count;
	float source_temperature;
	float temperature_flip_alpha;
	std::string visualization;
	virtual void initialize(const Config &config) {
		FLIPFluid::initialize(Config(config).set("initializer", "full"));

		temperature = Array(width, height);
		temperature_backup = Array(width, height);
		temperature_count = Array(width, height);

		buoyancy = config.get("bouyancy", 0.1f);
		conduction = config.get("conduction", 0.0f);
		visualization = config.get("visualization", "");
		source_temperature = config.get("source_temperature", ambient_temp);
		show_air_particles = config.get("show_air_particles", false);
		temperature_flip_alpha = config.get("temperature_flip_alpha", 0.97f);
		gravity = Vector2(0, 0.5f * height);
		ambient_temp = 200;

		for (auto &p : particles) {
			p.temperature = ambient_temp;
			if (show_air_particles) {
				p.show = true;
			}
			else {
				p.show = false;
			}
		}
	}
	virtual void simple_mark_cells() {
		cell_types = CellType::WATER;
	}
	void seed_particles(float delta_t) {
		for (int i = 0; i < 100; i++) {
			Vector2 pos((0.4f + 0.2f * rand()) * width, (0.1f + 0.1f * rand()) * height);
			Vector2 vel(0.0f, 0.0f);
			Particle p(pos, vel);
			p.temperature = source_temperature;
			particles.push_back(p);
		}
	}
	void apply_external_forces(float delta_t) {
		for (auto &p : particles) {
			p.velocity += buoyancy * delta_t * Vector2(0, 1) * (p.temperature - ambient_temp);
		}
	}
	virtual void step(float delta_t) {
		seed_particles(delta_t);
		FLIPFluid::step(delta_t);
	}
	virtual void substep(float delta_t) {
		apply_external_forces(delta_t);
		simple_mark_cells();
		rasterize();
		rasterize_temperature();
		extrapolate();
		temperature_backup = temperature;
		backup_velocity_field();
		apply_boundary_condition();
		project(delta_t);
		advect(delta_t);
		if (conduction > 0)
			diffuse_temperature(delta_t);
		resample_temperature(delta_t);
		t += delta_t;
	}
	virtual void voronoi_extrapolate(Array &val, const Array &weight) {
		NearestNeighbour2D voronoi;
		vector<Vector2> points;
		vector<float> values;
		for (auto ind : val.get_region()) {
			if (weight[ind] > 0) {
				points.push_back(Vector2(float(ind.i), float(ind.j)));
				values.push_back(val[ind]);
			}
		}
		voronoi.initialize(points);
		for (auto ind : val.get_region()) {
			if (weight[ind] == 0) {
				val[ind] = values[voronoi.query_index(Vector2(float(ind.i), float(ind.j)))];
			}
		}
	}
	virtual void extrapolate() {
		voronoi_extrapolate(u, u_count);
		voronoi_extrapolate(v, v_count);
		voronoi_extrapolate(temperature, temperature_count);
	}
	virtual void voronoi_rasterize() {
		NearestNeighbour2D voronoi[2];
		vector<Vector2> points[2];
		vector<float> values[2];
		for (int k = 0; k < 2; k++) {
			for (int i = 0; i < (int)particles.size(); i++) {
				points[k].push_back(particles[i].position);
				values[k].push_back(particles[i].velocity[k]);
			}
			voronoi[k].initialize(points[k]);
		}
		for (int i = 0; i < width + 1; i++) {
			for (int j = 0; j < height; j++) {
				u[i][j] = values[0][voronoi[0].query_index(Vector2((float)i, j + 0.5f))];
			}
		}
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height + 1; j++) {
				v[i][j] = values[1][voronoi[1].query_index(Vector2(i + 0.5f, (float)j))];
			}
		}
	}
	void rasterize_temperature() {
		temperature = 0;
		temperature_count = 0;
		for (auto &p : particles) {
			int x, y;
			x = (int)floor(p.position.x - 0.5);
			y = (int)floor(p.position.y - 0.5);
			for (int dx = 0; dx < 2; dx++) {
				for (int dy = 0; dy < 2; dy++) {
					int nx = x + dx, ny = y + dy;
					if (!temperature.inside(nx, ny)) {
						continue;
					}
					float weight = kernel(p.position - vec2(nx + 0.5f, ny + 0.5f));
					temperature[nx][ny] += weight * p.temperature;
					temperature_count[nx][ny] += weight;
				}
			}
		}
		for (auto ind : temperature.get_region()) {
			if (temperature_count[ind] > 0) {
				temperature[ind] /= temperature_count[ind];
			}
			else {
				// extrapolation...
			}
		}
	}
	void diffuse_temperature(float delta_t) {
		float exchange = conduction * delta_t;
		Array new_temperature = temperature;
		for (auto ind : temperature.get_region()) {
			for (auto d : neighbour4) {
				auto nei = ind.neighbour(d);
				if (temperature.inside(nei)) {
					float delta = exchange * temperature[nei];
					new_temperature[ind] += delta;
					new_temperature[nei] -= delta;
				}
			}
		}
		temperature = new_temperature;
	}
	virtual void show(ImageBuffer<Vector3> &buffer) {
		FLIPFluid::show(buffer);
		for (auto ind : temperature.get_region()) {
			buffer[ind.i][ind.j] = Vector3(temperature[ind] / 1000);
		}
		if (visualization == "temperature_only") {
			float temp_max = 1000.0f;
			for (int i = 0; i < buffer.get_width(); i++) {
				for (int j = 0; j < buffer.get_height(); j++) {
					float x = (i + 0.5f) / buffer.get_width();
					float y = (j + 0.5f) / buffer.get_height();
					buffer[i][j] = Vector3(temperature.sample_relative_coord(x, y) / temp_max);
				}
			}
		}
		/*
		for (auto ind : temperature.get_region()) {
			float m = length(Vector2(u[ind], v[ind]));
			buffer[200 + ind.i][ind.j] = Vector3(m * 0.01);
		}
		for (auto ind : temperature.get_region()) {
			float m = u[ind] - u[ind.neighbour(Vector2(0, 1))] - v[ind] + v[ind.neighbour(Vector2(1, 0))];
			buffer[100 + ind.i][ind.j] = (lerp(m * 0.05f + 0.5f, Vector3(1, 0, 0), Vector3(0, 0, 1)));
		}
		*/
	}
	void resample_temperature(float delta_t) {
		float alpha = powf(temperature_flip_alpha, delta_t / 0.01f);
		for (auto &p : particles) {
			p.temperature = alpha * (-temperature_backup.sample(p.position.x, p.position.y) + p.temperature) + temperature.sample(p.position.x, p.position.y);
		}
	}
};

class VoronoiFLIPSmoke : public FLIPSmoke {
	virtual void substep(float delta_t) {
		apply_external_forces(delta_t);
		simple_mark_cells();
		rasterize();
		rasterize_temperature();
		extrapolate();
		temperature_backup = temperature;
		backup_velocity_field();
		apply_boundary_condition();
		project(delta_t);
		advect(delta_t);
		diffuse_temperature(delta_t);
		resample_temperature(delta_t);
		t += delta_t;
	}
};

TC_NAMESPACE_END
