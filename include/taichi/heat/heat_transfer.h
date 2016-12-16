#pragma once

#include "renderer/scene_geometry.h"

#include <taichi/common/meta.h>
#include "renderer/sampler.h"

TC_NAMESPACE_BEGIN

void heat_demo(Config config);

class HeatTransferSimulation {
public:
	HeatTransferSimulation(std::shared_ptr<Scene> scene, Config config) {
		this->scene = scene;
		this->ray_intersection = create_instance<RayIntersection>(config.get_string("ray_intersection"));
		this->sg = std::make_shared<SceneGeometry>(scene, ray_intersection);
		delta_t = config.get_float("delta_t");
		samples = config.get_int("samples");
		diffuse_substeps = config.get_int("diffuse_substeps");
		cond = config.get_real("cond");
		grid_cell_size = config.get_real("grid_cell_size");
		sampler = create_instance<Sampler>(config.get_string("sampler"));
		simulation_method = config.get_string("simulation_method");
		build_voxel();
	}
	void step() {
		radiate();
		rasterize();
		for (int i = 0; i < diffuse_substeps; i++) {
			diffuse(delta_t / diffuse_substeps);
		}
		unrasterize();
	}
	void radiate() {
		scene->update_emission_cdf();
		for (int i = 0; i < samples; i++) {
			Photon p;
			scene->sample_photon(p, sampler->sample(0, photon_counter), delta_t, 1.0f / samples);
			transfer_photon(p, photon_counter);
			photon_counter++;
		}
	}
	void rasterize();
	void diffuse(real delta_t);
	void unrasterize();
	void transfer_photon(Photon p, long long index);

	void build_voxel() {
		memset(grid, -1, sizeof(grid));
		int count = 0;
		for (int i = 0; i < grid_dim; i++) {
			for (int j = 0; j < grid_dim; j++) {
				int last_k = -1;
				start[i][j] = grid_dim;
				for (int k = 0; k < grid_dim; k++) {
					Ray ray(grid_point_to_world(Vector3(i, j, k)), Vector3(0, 0, -1), 0);
					IntersectionInfo info = sg->query(ray);
					if (info.triangle_id == -1) {
						continue;
					}
					Mesh *mesh = scene->get_mesh_from_triangle_id(info.triangle_id);
					bool inside = !info.front && mesh->need_voxelization;
					if (inside) {
						static const Vector3 directions[]{ {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1} };
						float min_dist = 1e30f;
						for (auto &dir : directions) {
							Ray tri_ray(grid_point_to_world(Vector3(i, j, k)), dir, 0);
							IntersectionInfo info = sg->query(tri_ray);
							if (info.triangle_id != -1 && info.dist <= min_dist) {
								min_dist = info.dist;
								grid[i][j][k] = info.triangle_id;
							}
						}
						grid_depth[i][j][k] = min_dist;
						temperature[i][j][k] = mesh->initial_temperature;
						if (last_k != -1) {
							next[i][j][last_k] = k;
						}
						else {
							start[i][j] = k;
						}
						last_k = k;
						next[i][j][k] = grid_dim;
						count++;
					}
					else {
						grid[i][j][k] = -1;
					}
				}
			}
		}
		P(count);
		cout << "Total Vol: " << count * powf(grid_cell_size, 3.0f) << endl;
	}
	bool grid_inside_domain(int x, int y, int z) {
		return 0 <= x && x < grid_dim && 0 <= y && y < grid_dim && 0 <= z && z < grid_dim;
	}
	bool grid_inside_mesh(int x, int y, int z) {
		return grid_inside_domain(x, y, z) && grid[x][y][z] != -1;
	}
private:
	Vector3 grid_point_to_world(Vector3 grid_point) {
		return (grid_point - Vector3(grid_dim / 2)) * Vector3(grid_cell_size);
	}
	static const int grid_dim = 200;
	real cond;
	real grid_cell_size;
	int diffuse_substeps;
	int grid[grid_dim][grid_dim][grid_dim];
	int next[grid_dim][grid_dim][grid_dim];
	int start[grid_dim][grid_dim];
	float temperature[grid_dim][grid_dim][grid_dim];
	float temperature_tmp[grid_dim][grid_dim][grid_dim];
	float grid_depth[grid_dim][grid_dim][grid_dim];
	long long photon_counter;
	std::shared_ptr<Sampler> sampler;
	std::shared_ptr<SceneGeometry> sg;
	std::shared_ptr<Scene> scene;
	std::shared_ptr<RayIntersection> ray_intersection;
	real delta_t;
	int samples;
	string simulation_method;
};

TC_NAMESPACE_END

