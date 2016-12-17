#include <taichi/visual/surface_material.h>
#include <taichi/physics/physics_constants.h>
#include <taichi/math/discrete_sampler.h>

TC_NAMESPACE_BEGIN

    class TransparentMaterial : public SurfaceMaterial {
    protected:
        std::shared_ptr<SurfaceMaterial> nested;
        std::shared_ptr<Texture> mask;
    public:
        void initialize(const Config &config) override {
            mask = AssetManager::get_asset<Texture>(config.get_int("mask"));
            nested = AssetManager::get_asset<SurfaceMaterial>(config.get_int("nested"));
        }

        virtual bool is_index_matched() const override {
            return nested->is_index_matched();
        }

        virtual void sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir, Vector3 &f, real &pdf,
                            SurfaceEvent &event, const Vector2 &uv) const override {
            real alpha = mask->sample(uv).x;
            if (u < alpha) {
                out_dir = -in_dir;
                f = Vector3(1.0f) * abs(1.0f / in_dir.z);
                pdf = alpha;
                event = (int) SurfaceScatteringFlags::delta | (int) SurfaceScatteringFlags::index_matched;
            } else {
                u = (u - alpha) / (1 - alpha);
                nested->sample(in_dir, u, v, out_dir, f, pdf, event, uv);
                pdf *= 1 - alpha;
            }
        }

        real get_alpha(const Vector2 &uv) const {
            return mask->sample(uv).x;
        }

        virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            real alpha = get_alpha(uv);
            return (1 - alpha) * nested->probability_density(in, out, uv);
        };

        virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            return nested->evaluate_bsdf(in, out, uv);
        }

        virtual bool is_delta() const override {
            return nested->is_delta();
        }
    };

    TC_IMPLEMENTATION(SurfaceMaterial, TransparentMaterial, "transparent")

TC_NAMESPACE_END
