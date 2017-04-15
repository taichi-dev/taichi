/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/visual/scene.h>
#include <taichi/visual/surface_material.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

TC_NAMESPACE_BEGIN

void Mesh::initialize(const Config &config) {
    transform = Matrix4(1.0f);
    std::string filepath = config.get_string("filename");
    if (!filepath.empty())
        load_from_file(filepath);
}

void Mesh::load_from_file(const std::string &file_path) {
    std::string inputfile = file_path;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, inputfile.c_str());

    if (!err.empty()) { // `err` may contain warning message.
        std::cerr << err << std::endl;
    }

    assert_info(ret, "Loading " + file_path + " failed");

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];

            // Loop over vertices in the face.
            assert_info(fv == 3, "Only triangles supported...");
            int i = (int)vertices.size(), j = i + 1, k = i + 2;
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                float vx = attrib.vertices[3 * idx.vertex_index + 0];
                float vy = attrib.vertices[3 * idx.vertex_index + 1];
                float vz = attrib.vertices[3 * idx.vertex_index + 2];
                float nx = attrib.normals[3 * idx.normal_index + 0];
                float ny = attrib.normals[3 * idx.normal_index + 1];
                float nz = attrib.normals[3 * idx.normal_index + 2];
                float tx = 0.0f, ty = 0.0f;
                if (idx.texcoord_index != -1) {
                    tx = attrib.texcoords[2 * idx.texcoord_index + 0];
                    ty = attrib.texcoords[2 * idx.texcoord_index + 1];
                }
                vertices.push_back(Vector3(vx, vy, vz));
                normals.push_back(Vector3(nx, ny, nz));
                uvs.push_back(Vector2(tx, ty));
            }
            untransformed_triangles.push_back(Triangle(vertices[i], vertices[j], vertices[k],
                normals[i], normals[j], normals[k],
                uvs[i], uvs[j], uvs[k], i / 3));
            index_offset += fv;
        }
    }
}

void Mesh::set_material(std::shared_ptr<SurfaceMaterial> material) {
    this->material = material;
    if (material->is_emissive()) {
        this->emission_color = Vector3(material->get_importance(Vector2(0.5f, 0.5f))); //TODO
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
    inter.tri_coord.x = coord_u;
    inter.tri_coord.y = coord_u;
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
    // inter.material = mesh->material.get();
    Vector3 u = normalized(t.v[1] - t.v[0]);
    real sgn = inter.front ? 1.0f : -1.0f;
    Vector3 v = normalized(cross(sgn * inter.normal, u)); // Due to shading normal, we have to normalize here...
    inter.dt_du = t.get_duv(u);
    inter.dt_dv = t.get_duv(v);
    // TODO: ...
    u = normalized(cross(v, inter.normal));
    inter.to_world = Matrix3(u, v, inter.normal);
    inter.to_local = glm::transpose(inter.to_world);
    return inter;
}

void Scene::add_mesh(std::shared_ptr<Mesh> mesh) {
    meshes.push_back(*mesh);
}

void Scene::finalize_geometry() {
    int triangle_count = 0;
    for (auto &mesh : meshes) {
        triangle_id_start[&mesh] = triangle_count;
        auto sub = mesh.get_triangles();
        for (int i = 0; i < (int)sub.size(); i++) {
            sub[i].id = triangle_count + i;
        }
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
};

void Scene::finalize_lighting() {
    if (!emissive_triangles.empty()) {
        update_emission_cdf();
        update_light_emission_cdf();
    }
    else {
        envmap_sample_prob = 1.0f;
        assert_info(envmap != nullptr, "There should be light sources.");
    }
}

void Scene::finalize() {
    finalize_geometry();
    finalize_lighting();
}

TC_NAMESPACE_END
