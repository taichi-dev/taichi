#include <taichi/visual/surface_material.h>
#include <taichi/physics/physics_constants.h>
#include <taichi/math/discrete_sampler.h>
#include <taichi/common/asset_manager.h>

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
            f = Vector3(alpha) * abs(1.0f / in_dir.z);
            pdf = alpha;
            event = (int)SurfaceScatteringFlags::delta | (int)SurfaceScatteringFlags::index_matched;
        }
        else {
            u = (u - alpha) / (1 - alpha);
            nested->sample(in_dir, u, v, out_dir, f, pdf, event, uv);
            f *= 1 - alpha;
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
        real alpha = get_alpha(uv);
        return (1 - alpha) * nested->evaluate_bsdf(in, out, uv);
    }

    virtual bool is_delta() const override {
        return nested->is_delta();
    }
};

TC_IMPLEMENTATION(SurfaceMaterial, TransparentMaterial, "transparent")

class DiffuseTransmissiveMaterial : public SurfaceMaterial {
protected:
    std::shared_ptr<Texture> color_sampler;
public:
    void initialize(const Config &config) override {
        color_sampler = get_color_sampler(config, "diffuse");
    }

    Vector3 sample_direction(const Vector3 &in, real u, real v, const Vector2 &uv) const {
        if (abs(in.z) > 1 - eps) {
            Vector3 normal(0, 0, -sgn(in.z));
            return random_diffuse(normal, u, v);
        }
        else {
            // We do the following other than the above to ensure correlation for MCMC...
            if (u > v) {
                std::swap(u, v);
            }
            if (v < eps) {
                v = eps;
            }
            u /= v;
            real xz = v, y = sqrt(1 - v * v);
            real phi = u * 2.0f * pi;
            real r = v / sqrt(in.x * in.x + in.y * in.y), p = in.x * r, q = in.y * r;
            real c = cos(phi), s = sin(phi);
            return Vector3(p * c - q * s, q * c + p * s, -y * sgn(in.z));
        }
    }

    virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        if (in.z * out.z > -eps) {
            return 0;
        }
        return abs(out.z) / pi;
    }

    virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        auto color = color_sampler->sample3(uv);
        return (in.z * out.z < -eps ? 1.0f : 0.0f) * color * (1.0f / pi);
    }

    virtual void
        sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir, Vector3 &f, real &pdf,
            SurfaceEvent &event, const Vector2 &uv) const override {
        out_dir = sample_direction(in_dir, u, v, uv);
        f = evaluate_bsdf(in_dir, out_dir, uv);
        event = (int)SurfaceScatteringFlags::non_delta;
        pdf = std::abs(out_dir.z) / pi;
    }
};

TC_IMPLEMENTATION(SurfaceMaterial, DiffuseTransmissiveMaterial, "difftrans")

TC_NAMESPACE_END
