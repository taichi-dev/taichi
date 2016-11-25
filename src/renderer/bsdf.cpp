#include "bsdf.h"
#include "scene.h"
#include "surface_material.h"

TC_NAMESPACE_BEGIN

    BSDF::BSDF(std::shared_ptr<Scene> const &scene, const IntersectionInfo *inter) {
        world_to_local = Matrix3(inter->to_local);
        local_to_world = Matrix3(inter->to_world);
        geometry_normal = inter->geometry_normal;
        material = scene->get_mesh_from_triangle_id(inter->triangle_id)->material.get();
        uv = inter->uv;
    }

    BSDF::BSDF(std::shared_ptr<Scene> const &scene, int triangle_id) { // initialize for light triangle
        Triangle t = scene->get_triangle(triangle_id);
        Vector3 u = normalized(t.v[1] - t.v[0]);
        float sgn = 1;
        Vector3 v = cross(sgn * t.normal, u);
        local_to_world = Matrix3(u, v, t.normal);
        world_to_local = glm::transpose(local_to_world);
        geometry_normal = t.normal;
        material = scene->get_mesh_from_triangle_id(triangle_id)->material.get();
        uv = Vector2(0.5f);
    }

    Vector3 BSDF::sample_direction(const Vector3 &in, real u, real v) const {
        Vector3 out = local_to_world * material->sample_direction(world_to_local * in, u, v, uv);
        return out;
    }

    void BSDF::sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir,
                      Vector3 &f, real &pdf, SurfaceMaterial::ScatteringEvent &event) const {
        const Vector3 in_dir_local = world_to_local * in_dir;
        Vector3 out_dir_local;
        material->sample(in_dir_local, u, v, out_dir_local, f, pdf, event, uv);
        out_dir = local_to_world * out_dir_local;
    }

    real BSDF::probability_density(const Vector3 &in, const Vector3 &out) const {
        return material->probability_density(world_to_local * in, world_to_local * out, uv);
    }

    Vector3 BSDF::evaluate(const Vector3 &in, const Vector3 &out) const {
        if (dot(geometry_normal, out) * (world_to_local * out).z < 0.0f) {
            // for shaded/interpolated normal consistency
            return Vector3(0.0f);
        }
        Vector3 output = material->evaluate_bsdf(world_to_local * in, world_to_local * out, uv);
        assert_info(output.r >= 0 && output.g >= 0 && output.b >= 0, "BSDF should not be negative.");
        return output;
    }

    bool BSDF::is_delta() const {
        assert_info(material != nullptr, "material is empty!");
        return material->is_delta();
    }

    bool BSDF::is_emissive() const {
        assert_info(material != nullptr, "material is empty!");
        return material->is_emissive();
    }

    std::string BSDF::get_name() const {
        return material->get_name();
    }

TC_NAMESPACE_END
