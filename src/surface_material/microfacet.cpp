#include <taichi/visual/surface_material.h>

TC_NAMESPACE_BEGIN

// Disney "principled" BRDF
// https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf

// GGX, a.k.a. Trowbridge-Reitz, described in the paper "Microfacet Models for Refraction through Rough Surfaces"

class MicrofacetMaterial final : public SurfaceMaterial {
protected:
    std::shared_ptr<Texture> color_sampler;
    std::shared_ptr<Texture> roughness_sampler;
    real f0;

public:
    void initialize(const Config &config) override {
        color_sampler = get_color_sampler(config, "color");
        assert(color_sampler != nullptr);
        roughness_sampler = get_color_sampler(config, "roughness");
        // m = sqrt(2/(a + 2)) for Phong
        assert(roughness_sampler != nullptr);
        f0 = config.get_real("f0");
    }

    // D can be GGX, Beckmann, Blinn-Phong
    real evaluateD(real roughness, const Vector3 &h) const {
        const real cos_t = h.z;
        return sqr(roughness) / std::max(1e-6f, (pi * sqr((sqr(roughness) - 1) * sqr(cos_t) + 1.0f)));
    }

    Vector3 sampleD(real roughness, const Vector2 &p) const {
        const real phi = p.x * 2 * pi;
        const real theta = std::acos(std::sqrt((1 - p.y) / (1 + p.y * (roughness * roughness - 1))));
        return Vector3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    }

    // Fresnel term: Schlick approx.
    real F(real f0, real cos_theta) const {
        real c = 1 - cos_theta;
        return f0 + (1 - f0) * ((c * c) * (c * c) * c);
    }

    // Shadowing term for GGX only
    real G(real roughness, const Vector3 &in_dir, const Vector3 &out_dir, const Vector3 &h) const {
        if (dot(in_dir, h) * in_dir.z < eps) {
            return 0.0f;
        }
        const real a = 0.5f + roughness * 0.5f;
        return 2.0f / (1 + std::sqrt(1 + a * a * (sqr(1.0f / std::max(1e-6f, std::abs(in_dir.z))) - 1.0f)));
    }

    static Vector3 reflect(const Vector3 &in, const Vector3 &_h) {
        auto h = normalized(_h);
        return in - 2.0f * (in - dot(in, h) * h);
    }

    Vector3 sample_direction(const Vector3 &in, real u, real v, const Vector2 &uv) const {
        return reflect(in, sampleD(get_roughness(uv), Vector2(u, v)));
        // The result is guarded by evaluate_brdf form penetration
    }

    real get_roughness(const Vector2 &uv) const {
        return std::max(1e-3f, roughness_sampler->sample(uv).x);
    }

    real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        if (in.z * out.z < eps) {
            return 0;
        }
        const Vector3 h = normalized(in + out);
        return std::abs(evaluateD(get_roughness(uv), h) * h.z / std::max(1e-6f, 4.0f * dot(out, h)));
    }

    Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        auto color = color_sampler->sample3(uv);
        if (in.z * out.z < eps) {
            return Vector3(0.0f);
        }
        real roughness = get_roughness(uv);
        const Vector3 h = normalized(in + out);
        real factor = F(f0, std::max(0.0f, dot(in, h))) * G(roughness, in, out, h) * evaluateD(roughness, h);
        factor *= 1.0f / (4.0f * std::max(1e-5f, std::abs(in.z)) * std::abs(out.z));
        return color * factor;
    }

    void sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir,
        Vector3 &f, real &pdf,
        SurfaceEvent &event, const Vector2 &uv) const override {
        out_dir = sample_direction(in_dir, u, v, uv);
        f = evaluate_bsdf(in_dir, out_dir, uv);
        event = (int)SurfaceScatteringFlags::non_delta;
        pdf = probability_density(in_dir, out_dir, uv);
    }

    real get_importance(const Vector2 &uv) const override {
        return luminance(color_sampler->sample3(uv));
    }
};

TC_IMPLEMENTATION(SurfaceMaterial, MicrofacetMaterial, "microfacet");

TC_NAMESPACE_END
