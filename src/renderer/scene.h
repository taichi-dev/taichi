#pragma once

#include "geometry_primitives.h"
#include "io/importer.h"
#include "camera.h"
#include <map>
#include <deque>
#include "physics/physics_constants.h"
#include "physics/spectrum.h"
#include "discrete_sampler.h"
#include "surface_material.h"
#include "envmap.h"
#include "volume.h"

TC_NAMESPACE_BEGIN

struct Photon {
	Vector3 pos, dir;
	real energy;
};

struct Face {
	Face() { }

	Face(int v0, int v1, int v2) {
		vert_ind[0] = v0;
		vert_ind[1] = v1;
		vert_ind[2] = v2;
	}

	Face(string dat) {
		std::stringstream ss(dat);
		for (int i = 0; i < 3; i++) {
			ss >> vert_ind[i];
		}
	}

	int vert_ind[3];
	int material;
};

class Mesh {
public:
	Mesh(ptree &pt);
	Mesh() {}

	void initialize(const Config &config);
	void set_material(std::shared_ptr<SurfaceMaterial> material);
	void load_from_file(const std::string &file_path);
	void translate(const Vector3 &offset);
	void scale(const Vector3 &scales);
	void scale_s(real scale);
	void rotate_euler(const Vector3 &euler_angles);
	void rotate_angle_axis(real angle, const Vector3 &axis);
	std::vector<Triangle> get_triangles(int triangle_count) {
		std::vector<Triangle> triangles;
		for (auto face : faces) {
			auto t = Triangle(
				multiply_matrix4(transform, vertices[face.vert_ind[0]], 1.0f),
				multiply_matrix4(transform, vertices[face.vert_ind[1]], 1.0f),
				multiply_matrix4(transform, vertices[face.vert_ind[2]], 1.0f),
				multiply_matrix4(transform, normals[face.vert_ind[0]], 0.0f),
				multiply_matrix4(transform, normals[face.vert_ind[1]], 0.0f),
				multiply_matrix4(transform, normals[face.vert_ind[2]], 0.0f),
				uvs[face.vert_ind[0]],
				uvs[face.vert_ind[1]],
				uvs[face.vert_ind[2]],
				triangle_count++);
			triangles.push_back(t);
		}
		return triangles;
		/*
		vector<Triangle> triangles;
		for (auto face : faces) {
			deque<Triangle> sub_divs;
			sub_divs.push_back(Triangle(
				vertices[face.vert_ind[0]],
				vertices[face.vert_ind[1]],
				vertices[face.vert_ind[2]],
				0
				));
			if (sub_div_limit > 0) {
				while (true) {
					Triangle t = sub_divs.front();
					int i;
					if (t.max_edge_length(i) < sub_div_limit)
						break;
					Vector3 mid = 0.5f * (t.v[i] + t.v[(i + 1) % 3]);
					sub_divs.pop_front();
					sub_divs.push_back(Triangle(t.v[i], mid, t.v[(i + 2) % 3], 0));
					sub_divs.push_back(Triangle(t.v[(i + 2) % 3], mid, t.v[(i + 1) % 3], 0));
				}
			}
			for (auto &t : sub_divs) {
				push_triangle(
					t.v[0],
					t.v[1],
					t.v[2],
					triangle_count,
					triangles
					);
			}
		}*/
	}

	bool need_voxelization;
	std::vector<Vector3> vertices;
	std::vector<Vector3> normals;
	std::vector<Vector2> uvs;
	real initial_temperature;
	std::vector<Face> faces;
	Matrix4 transform;
	real emission;
	Vector3 color;
	bool const_temp;
	real sub_div_limit;
	Vector3 emission_color;
	std::shared_ptr<SurfaceMaterial> material;
};

struct IntersectionInfo {
	IntersectionInfo() {
		triangle_id = -1;
		intersected = false;
	}

	bool front;
	bool intersected;
	Vector3 pos, normal, geometry_normal;
	Vector2 uv;
	Vector3 color;
	SurfaceMaterial *material = nullptr;
	int triangle_id;
	Matrix3 to_local;
	Matrix3 to_world;
	float dist;
};

class Scene {
public:
	Scene() {
	}
	Scene(ptree &pt, real sub_divide_limit) {
		this->sub_divide_limit = sub_divide_limit;
		load(pt);
	}

	void load(ptree &pt);

	void set_envmap(std::shared_ptr<EnvironmentMap> envmap) {
		this->envmap = envmap;
	}

	void set_atmosphere_material(std::shared_ptr<VolumeMaterial> vol_mat) {
		this->atmosphere_material = vol_mat;
	}

	std::shared_ptr<VolumeMaterial> get_atmosphere_material() const {
		return this->atmosphere_material;
	}

	void add_mesh(std::shared_ptr<Mesh> mesh);

	void finalize();

	void update_light_emission_cdf() {
		light_total_emission = 0;
		light_total_area = 0;
		std::vector<real> emissions;
		for (auto tri : emissive_triangles) {
			real e = tri.area * get_mesh_from_triangle_id(tri.id)->emission;
			light_total_emission += e;
			light_total_area += tri.area;
			emissions.push_back(e);
		}
		light_emission_sampler.initialize(emissions);
	}

	real get_average_emission() const {
		return light_total_emission / light_total_area;
	}

	void update_emission_cdf() {
		emission_cdf.clear();
		total_emission = 0;
		for (auto tri : triangles) {
			real e = tri.area * pow(tri.temperature, 4.0f);
			emission_cdf.push_back(e);
			total_emission += e;
		}
		real inv_tot = 1.0f / total_emission;
		for (int i = 0; i < (int)emission_cdf.size() - 1; i++) {
			emission_cdf[i + 1] += emission_cdf[i];
		}
		emission_cdf.push_back(1);
		for (auto &e : emission_cdf) {
			e *= inv_tot;
		}
	}

	std::vector<Triangle> &get_triangles() {
		return triangles;
	}

	Triangle get_triangle(int id) const {
		return triangles[id];
	}

	IntersectionInfo get_intersection_info(int triangle_id, Ray &ray);

	Triangle &sample_triangle_light_emission(real r, real &pdf) {
		int e_tid = light_emission_sampler.sample(r, pdf);
		return emissive_triangles[e_tid];
	}

	real get_triangle_pdf(int id) const {
		return triangles[id].area * get_mesh_from_triangle_id(id)->emission / light_total_emission;
	}

	void sample_photon(Photon &p, real r, real delta_t, real weight) {
		int tid = min(int(std::lower_bound(emission_cdf.begin(), emission_cdf.end(), r) - emission_cdf.begin()),
			(int)triangles.size() - 1);
		Triangle &t = triangles[tid];
		Mesh *mesh = triangle_id_to_mesh[tid];
		weight = t.area / total_triangle_area;
		p.dir = random_diffuse(t.normal);
		p.pos = t.sample_point();
		p.energy = weight * total_emission * delta_t * stefan_boltzmann_constant;
		if (!mesh->const_temp) {
			t.temperature -= p.energy / t.heat_capacity;
			t.temperature = max(t.temperature, 0.0f);
		}
	}

	void recieve_photon(int triangle_id, real energy) {
		int tid = triangle_id;
		Triangle &t = triangles[tid];
		Mesh *mesh = triangle_id_to_mesh[tid];
		if (!mesh->const_temp)
			t.temperature += energy / t.heat_capacity;
	}

	real get_temperature(int triangle_id, real u, real v) {
		error("not implemented");
		return 0.0f;
		/*
		real cooef[3]{ 1 - u - v, u, v };
		Mesh *mesh = triangle_id_to_mesh[triangle_id];
		real temp = 0;
		for (int i = 0; i < 3; i++) {
			int vertice_index = mesh->faces[triangle_id - triangle_id_start[mesh]].vert_ind[i];
			temp += mesh->temperature[vertice_index] * cooef[i];
		}
		return temp;
		*/
	}

	Vector3 get_coord(int triangle_id, real u, real v) const {
		real cooef[3]{ 1 - u - v, u, v };
		Mesh *mesh = triangle_id_to_mesh.find(triangle_id)->second;
		Vector3 temp(0);
		for (int i = 0; i < 3; i++) {
			int vertice_index = mesh->faces[triangle_id - triangle_id_start.find(mesh)->second].vert_ind[i];
			temp += mesh->vertices[vertice_index] * cooef[i];
		}
		return temp;
	}

	Mesh *get_mesh_from_triangle_id(int triangle_id) const {
		return triangle_id_to_mesh.find(triangle_id)->second;
	}

	real get_triangle_emission(int triangle_id) const {
		return get_mesh_from_triangle_id(triangle_id)->emission;
	}

	DiscreteSampler light_emission_sampler;
	std::shared_ptr<Camera> camera;
	std::vector<Triangle> triangles;
	std::vector<Triangle> emissive_triangles;
	std::vector<real> emission_cdf;
	real total_emission;
	real light_total_emission;
	real light_total_area;
	std::vector<Mesh> meshes;
	std::map<int, Mesh *> triangle_id_to_mesh;
	std::map<Mesh *, int> triangle_id_start;
	int num_triangles;
	real sub_divide_limit;
	real total_triangle_area;
	int resolution_x, resolution_y;
	std::shared_ptr<VolumeMaterial> atmosphere_material;
	std::shared_ptr<EnvironmentMap> envmap;
};

TC_NAMESPACE_END

