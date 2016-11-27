#pragma once

#include "math/linalg.h"
#include "renderer/scene.h"

TC_NAMESPACE_BEGIN

    class SurfaceMaterial;
    enum class SurfaceScatteringFlags;
    class Scene;
    struct IntersectionInfo;

    // Generalized BSDF (supposed to include photon and importon source)
    class BSDF {
    protected:
        SurfaceMaterial *material;
        Matrix3 world_to_local; // shaded normal
        Matrix3 local_to_world; // shaded normal
        Vector3 geometry_normal;
        Vector2 uv;

    public:
        BSDF() {
            material = nullptr;
        }
        BSDF(std::shared_ptr<Scene> const &scene, const IntersectionInfo &inter);
        BSDF(std::shared_ptr<Scene> const &scene, int triangle_id); // initialize for light triangle
        real cos_theta(const Vector3 &out) {
            return abs((world_to_local * out).z);
        }
        Vector3 sample_direction(const Vector3 &in, real u, real v) const;
        void sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir,
                       Vector3 &f, real &pdf, SurfaceEvent&event) const;
        real probability_density(const Vector3 &in, const Vector3 &out) const;
        Vector3 get_geometry_normal() {
            return geometry_normal;
        }
        Vector3 evaluate(const Vector3 &in, const Vector3 &out) const;
        bool is_delta() const;
        bool is_emissive() const;
		bool is_index_matched() const {
			return material->is_index_matched();
		}
		bool is_entering(const Vector3 &in_dir) const {
			return glm::dot(geometry_normal, in_dir) > 0;
		}
		VolumeMaterial const *get_internal_material() const {
			return material->get_internal_material();
		}
        std::string get_name() {return "TBD";};
    };

TC_NAMESPACE_END
