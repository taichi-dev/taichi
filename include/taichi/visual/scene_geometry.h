#pragma once

#include "ray_intersection.h"
#include "scene.h"

TC_NAMESPACE_BEGIN

// Generates Intersection Information for scene using ray_intersection
class SceneGeometry {
public:
    SceneGeometry(std::shared_ptr<Scene> scene, std::shared_ptr<RayIntersection> ray_intersection) {
        this->scene = scene;
        this->ray_intersection = ray_intersection;
        for (auto &tri : scene->get_triangles()) {
            ray_intersection->add_triangle(tri);
        }
        rebuild();
    }

    void rebuild() {
        ray_intersection->build();
    }

    int query_hit_triangle_id(Ray &ray) {
        ray_intersection->query(ray);
        return ray.triangle_id;
    }

    IntersectionInfo query(Ray &ray) {
        int tri_id = query_hit_triangle_id(ray);
        return scene->get_intersection_info(tri_id, ray);
    }

    IntersectionInfo occlude(Ray &ray) {
        int tri_id = query_hit_triangle_id(ray);
        return scene->get_intersection_info(tri_id, ray);
    }

private:
    std::shared_ptr<Scene> scene;
    std::shared_ptr<RayIntersection> ray_intersection;
};

TC_NAMESPACE_END

