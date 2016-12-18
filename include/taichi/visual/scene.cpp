#include "scene.h"
#include "surface_material.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

TC_NAMESPACE_BEGIN

Assimp::Importer importer;
void Mesh::translate(const Vector3 &offset) {
	transform = glm::translate(transform, offset);
}
void Mesh::scale(const Vector3 &scales) {
	transform = glm::scale(transform, scales);
}
void Mesh::scale_s(real scale) {
	transform = glm::scale(transform, Vector3(scale));
}
void Mesh::rotate_euler(const Vector3 &euler_angles) {
	rotate_angle_axis(euler_angles.x, Vector3(1.0f, 0.0f, 0.0f));
	rotate_angle_axis(euler_angles.y, Vector3(0.0f, 1.0f, 0.0f));
	rotate_angle_axis(euler_angles.z, Vector3(0.0f, 0.0f, 1.0f));
}
void Mesh::rotate_angle_axis(real angle, const Vector3 &axis) {
	transform = glm::rotate(transform, angle * pi / 180.0f, axis);
}

void Mesh::initialize(const Config &config) {
	transform = Matrix4(1.0f);
	std::string filepath = config.get_string("filename");
	load_from_file(filepath);
}

void Mesh::load_from_file(const std::string &file_path) {
	auto scene = importer.ReadFile(file_path,
		aiProcess_Triangulate |
		aiProcess_JoinIdenticalVertices |
		aiProcess_SortByPType);
	assert_info(scene != nullptr, std::string("Mesh file ") + file_path + " load failed");
	for (int mesh_index = 0; mesh_index < (int)scene->mNumMeshes; mesh_index++) {
		aiMesh *mesh = scene->mMeshes[mesh_index];
		for (int face_index = 0; face_index < (int)mesh->mNumFaces; face_index++) {
			aiFace *face = &mesh->mFaces[face_index];
			faces.push_back(Face((int)vertices.size(), (int)vertices.size() + 1, (int)vertices.size() + 2));
			for (int i = 0; i < 3; i++) {
				auto conv = [](aiVector3D vec) {
					return Vector3(vec.x, vec.y, vec.z);
				};
				auto conv2 = [](aiVector3D vec) {
					return Vector2(vec.x, vec.y);
				};
				Vector3 normal = conv(mesh->mNormals[face->mIndices[i]]);
				Vector3 vertex = conv(mesh->mVertices[face->mIndices[i]]);
				Vector2 uv(0.0f);
				if (mesh->GetNumUVChannels() >= 1)
					uv = conv2(mesh->mTextureCoords[0][face->mIndices[i]]);
				normals.push_back(normal);
				vertices.push_back(vertex);
				uvs.push_back(uv);
			}
		}
	}
}

void Mesh::set_material(std::shared_ptr<SurfaceMaterial> material) {
	this->material = material;
	if (material->is_emissive()) {
		this->emission_color = Vector3(material->get_intensity(Vector2(0.5f, 0.5f))); //TODO
		this->emission = luminance(this->emission_color);
	}
	else {
		this->emission = 0.0f;
		this->emission_color = Vector3(0);
	}
}

IntersectionInfo Scene::get_intersection_info(int triangle_id, Ray &ray) {
	IntersectionInfo inter;
	if (triangle_id == -1) {
		return inter;
	}
	inter.intersected = true;
	Triangle &t = triangles[triangle_id];
	real coord_u = ray.u, coord_v = ray.v;
	inter.pos = t.v[0] + coord_u * (t.v[1] - t.v[0]) + coord_v * (t.v[2] - t.v[0]);
	inter.front = dot(ray.orig - t.v[0], t.normal) > 0;
	// Verify interpolated normals can lead specular rays to go inside the object.
	Vector3 normal = t.get_normal(coord_u, coord_v);
	Vector2 uv = t.get_uv(coord_u, coord_v);
	inter.uv = uv;
	inter.geometry_normal = inter.front ? t.normal : -t.normal;
	inter.normal = inter.front ? normal : -normal;
	Mesh *mesh = triangle_id_to_mesh[t.id];
	inter.triangle_id = triangle_id;
	inter.dist = ray.dist;
	inter.material = mesh->material.get();
	Vector3 u = normalized(t.v[1] - t.v[0]);
	real sgn = inter.front ? 1.0f : -1.0f;
	Vector3 v = normalized(cross(sgn * inter.normal, u)); // Due to shading normal, we have to normalize here...
	// TODO: ...
	u = normalized(cross(v, inter.normal));
	inter.to_world = Matrix3(u, v, inter.normal);
	inter.to_local = glm::transpose(inter.to_world);
	return inter;
}

void Scene::add_mesh(std::shared_ptr<Mesh> mesh) {
	meshes.push_back(*mesh);
}

void Scene::finalize() {
	int triangle_count = 0;
	for (auto &mesh : meshes) {
		triangle_id_start[&mesh] = triangle_count;
		auto sub = mesh.get_triangles(triangle_count);
		triangle_count += (int)sub.size();
		triangles.insert(triangles.end(), sub.begin(), sub.end());
		if (mesh.emission > 0) {
			emissive_triangles.insert(emissive_triangles.end(), sub.begin(), sub.end());
		}
		for (auto &tri : sub) {
			triangle_id_to_mesh[tri.id] = &mesh;
		}
	}
	num_triangles = triangle_count;
	printf("Scene loaded. Triangle count: %d\n", triangle_count);
	if (!emissive_triangles.empty()) {
		update_emission_cdf();
		update_light_emission_cdf();
	}
	else {
		envmap_sample_prob = 1.0f;
		assert_info(envmap != nullptr, "There should be light sources.");
	}
}

TC_NAMESPACE_END
