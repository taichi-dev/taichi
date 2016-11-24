#include "scene.h"
#include "material.h"
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
        for (int mesh_index = 0; mesh_index < (int) scene->mNumMeshes; mesh_index++) {
            aiMesh *mesh = scene->mMeshes[mesh_index];
            for (int face_index = 0; face_index < (int) mesh->mNumFaces; face_index++) {
                aiFace *face = &mesh->mFaces[face_index];
                faces.push_back(Face((int) vertices.size(), (int) vertices.size() + 1, (int) vertices.size() + 2));
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

	void Mesh::set_material(std::shared_ptr<Material> material) {
		this->material = material;
		if (material->is_emissive()) {
			this->emission_color = Vector3(1); //TODO
            this->emission = luminance(this->emission_color);
		}
		else {
			this->emission = 0.0f;
			this->emission_color = Vector3(0);
		}
	}

    Mesh::Mesh(ptree &pt) {
        transform = glm::mat4(1.0f);
        std::string filepath = pt.get<std::string>("filepath");
		load_from_file(filepath);
        initial_temperature = pt.get("temperature", 0.0f);
        need_voxelization = pt.get("need_voxelization", 0.0f) > 0;
        const_temp = pt.get("const_temp", 0.0f) > 0;
        sub_div_limit = pt.get("sub_divide_limit", 0.0f);

        emission = 0.0f;
        auto material_node = pt.get_child("material");
        std::string material_type = material_node.get<std::string>("type");
        if (material_type == "light_source") {
            this->emission_color = load_vector3(material_node.get("emission_color", std::string("(0.5, 0.5, 0.5)")));
            this->emission = luminance(this->emission_color);
            material = create_instance<Material>("emissive");
            material->initialize(Config().set("color", emission_color));
            material->set_color_sampler(std::make_shared<ConstantTexture>(emission_color));
        } else if (material_type == "pbr") {
            material = create_instance<Material>("pbr");
            material->initialize(material_node);
        } else {
            error("No material found.");
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

    void Scene::load(ptree &pt) {
        int triangle_count = 0;
                foreach(ptree::value_type & v, pt.get_child("scene.meshes")) {
                        v.second.add("sub_divide_limit", this->sub_divide_limit);
                        meshes.push_back(Mesh(v.second));
                    }
        for (auto &mesh : meshes) {
            triangle_id_start[&mesh] = triangle_count;
            auto sub = mesh.get_triangles(triangle_count);
            triangle_count += (int) sub.size();
            triangles.insert(triangles.end(), sub.begin(), sub.end());
            if (mesh.emission > 0) {
                emissive_triangles.insert(emissive_triangles.end(), sub.begin(), sub.end());
            }
            for (auto &tri : sub) {
                triangle_id_to_mesh[tri.id] = &mesh;
            }
        }
        resolution_x = pt.get<int>("scene.render.resolution_x");
        resolution_y = pt.get<int>("scene.render.resolution_y");
		/*
        Config camera_config;
		Matrix4 t = (load_matrix4(pt.get_child("scene.camera.transform")));
		P(t);
		camera_config.set("origin", multiply_matrix4(t, Vector3(0.0f, 0.0f, 1.0f), 1.0f))
			.set("look_at", multiply_matrix4(t, Vector3(0.0f, 0.0f, 0.0f), 1.0f))
			.set("up", multiply_matrix4(t, Vector3(0.0f, 0.0f, 1.0f), 0.0f))
			.set("fov_angle", 60.0f)
			.set("aspect_ratio", 1);// (real)resolution_x / resolution_y);
		*/
        camera = create_instance<Camera>("perspective");
        camera->initialize(pt.get_child("scene.camera"), (real)resolution_x / resolution_y);
        num_triangles = triangle_count;
        printf("Scene loaded. Triangle count: %d\n", triangle_count);
        total_triangle_area = 0.0f;
        for (auto tri : triangles) {
            total_triangle_area += tri.area;
        }
        update_emission_cdf();
        update_light_emission_cdf();
    }

	void Scene::add_mesh(std::shared_ptr<Mesh> mesh) {
		meshes.push_back(*mesh);
	}

	void Scene::finalize() {
        int triangle_count = 0;
        for (auto &mesh : meshes) {
            triangle_id_start[&mesh] = triangle_count;
            auto sub = mesh.get_triangles(triangle_count);
            triangle_count += (int) sub.size();
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
        update_emission_cdf();
        update_light_emission_cdf();
	}

TC_NAMESPACE_END
