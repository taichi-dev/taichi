#include <taichi/visual/surface_material.h>
#include <taichi/physics/physics_constants.h>
#include <taichi/math/discrete_sampler.h>

TC_NAMESPACE_BEGIN

TC_INTERFACE_DEF(SurfaceMaterial, "material");

class EmissiveMaterial : public SurfaceMaterial {
protected:
    std::shared_ptr<Texture> color_sampler;
public:
    virtual void initialize(const Config &config) override {
        SurfaceMaterial::initialize(config);
        color_sampler = get_color_sampler(config, "color");
    }

    virtual bool is_emissive() const override {
        return true;
    }

    Vector3 sample_direction(const Vector3 &in, real u, real v, const Vector2 &uv) const {
        return random_diffuse(Vector3(0, 0, in.z > 0 ? 1 : -1), u, v);
    }

    virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        if (in.z * out.z < eps) {
            return 0;
        }
        return out.z / pi;
    }

    virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        auto color = color_sampler->sample3(uv);
        return (in.z * out.z > 0 ? 1.0f : 0.0f) * color; // No division by pi here.
    }

    virtual real get_importance(const Vector2 &uv) const override {
        return luminance(color_sampler->sample3(uv));
    }

};

TC_IMPLEMENTATION(SurfaceMaterial, EmissiveMaterial, "emissive");

class DiffuseMaterial : public SurfaceMaterial {
protected:
    std::shared_ptr<Texture> color_sampler;
public:
    void initialize(const Config &config) override {
        color_sampler = get_color_sampler(config, "color");
        assert(color_sampler != nullptr);
    }

    Vector3 sample_direction(const Vector3 &in, real u, real v, const Vector2 &uv) const {
        Vector3 normal(0, 0, sgn(in.z));
        if (abs(in.z) > 1 - eps) {
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
            return Vector3(p * c - q * s, q * c + p * s, y * sgn(in.z));
        }
    }

    virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        if (in.z * out.z < eps) {
            return 0;
        }
        return std::abs(out.z) / pi;
    }

    virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        auto color = color_sampler->sample3(uv);
        return (in.z * out.z > eps ? 1.0f : 0.0f) * color * (1.0f / pi);
    }

    virtual void sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir,
        Vector3 &f, real &pdf,
        SurfaceEvent &event, const Vector2 &uv) const override {
        out_dir = sample_direction(in_dir, u, v, uv);
        f = evaluate_bsdf(in_dir, out_dir, uv);
        event = (int)SurfaceScatteringFlags::non_delta;
        pdf = out_dir.z / pi;
    }

    virtual real get_importance(const Vector2 &uv) const override {
        return luminance(color_sampler->sample3(uv));
    }
};

TC_IMPLEMENTATION(SurfaceMaterial, DiffuseMaterial, "diffuse");

class GlossyMaterial : public SurfaceMaterial {
protected:
    std::shared_ptr<Texture> color_sampler;
    std::shared_ptr<Texture> glossiness_sampler;
public:

    void initialize(const Config &config) override {
        color_sampler = get_color_sampler(config, "color");
        glossiness_sampler = get_color_sampler(config, "glossiness");
        assert(color_sampler != nullptr);
        assert(glossiness_sampler != nullptr);
    }

    Vector3 sample_direction(const Vector3 &in, real u, real v, const Vector2 &uv) const {
        real glossiness = glossiness_sampler->sample(uv).x;
        const Vector3 r = reflect(in);
        const Vector3 p = r.z > 1 - 1e-5f ? Vector3(0, 1, 0) : normalized(cross(Vector3(0, 0, 1), r));
        const Vector3 q = normalized(cross(r, p));
        const real phi = 2.0f * pi * u, d = pow(v, 1.0f / (glossiness + 1.0f));
        const real s = sin(phi), c = cos(phi), t = sqrt(1.0f - d * d);
        return (s * p + c * q) * t + d * r;
    }

    virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        real glossiness = glossiness_sampler->sample(uv).x;
        if (in.z * out.z < eps) {
            return 0;
        }
        const Vector3 r = reflect(in);
        return (glossiness + 1.0f) / (2.0f * pi) * pow(max(dot(r, out), 0.0f), glossiness);
    }

    virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        real glossiness = glossiness_sampler->sample(uv).x;
        if (in.z * out.z < eps) {
            return Vector3(0);
        }
        const Vector3 r = reflect(in);
        real t = std::min(std::max(dot(r, out), 0.0f), 1.0f);
        auto color = color_sampler->sample3(uv);
        return color * (glossiness + 2.0f) / (2.0f * pi) * pow(t, glossiness)
            / std::max(std::max(std::abs(in.z), std::abs(out.z)), 1e-7f);
    }

    virtual void sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir,
        Vector3 &f, real &pdf, SurfaceEvent &event, const Vector2 &uv) const override {
        out_dir = sample_direction(in_dir, u, v, uv);
        f = evaluate_bsdf(in_dir, out_dir, uv);
        event = (int)SurfaceScatteringFlags::non_delta;
        pdf = probability_density(in_dir, out_dir, uv);
    }

    virtual real get_importance(const Vector2 &uv) const override {
        return luminance(color_sampler->sample3(uv));
    }
};

TC_IMPLEMENTATION(SurfaceMaterial, GlossyMaterial, "glossy");

class ReflectiveMaterial : public SurfaceMaterial {
protected:
    std::shared_ptr<Texture> color_sampler;
public:
    void initialize(const Config &config) override {
        color_sampler = get_color_sampler(config, "color");
        assert(color_sampler != nullptr);
    }

    virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        return 1;
    };

    virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        return Vector3(0.0f);
    };

    virtual bool is_delta() const override {
        return true;
    }

    virtual void sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir,
        Vector3 &f, real &pdf, SurfaceEvent &event, const Vector2 &uv) const override {
        out_dir = reflect(in_dir);
        auto color = color_sampler->sample3(uv);
        f = color * std::abs(1.0f / std::max(0.0f, out_dir.z));
        event = (int)SurfaceScatteringFlags::delta;
        pdf = probability_density(in_dir, out_dir, uv);
    }

    virtual real get_importance(const Vector2 &uv) const override {
        return luminance(color_sampler->sample3(uv));
    }
};

TC_IMPLEMENTATION(SurfaceMaterial, ReflectiveMaterial, "reflective");

class RefractiveMaterial : public SurfaceMaterial {
    // See: https://en.wikipedia.org/wiki/Fresnel_equations

protected:
    real inside_ior = 1.5f;
    real outside_ior = 1.0f;
    std::shared_ptr<Texture> color_sampler;

public:
    void initialize(const Config &config) override {
        inside_ior = config.get_real("ior");
        color_sampler = get_color_sampler(config, "color");
        assert(color_sampler != nullptr);
    }

    real get_refraction(const Vector3 &in, Vector3 &out_reflect, Vector3 &out_refract) const {
        // returns refraction probability
        out_reflect = reflect(in);
        bool into = in.z > 0;
        real ior = get_ior(in);
        real cos_in = abs(in.z);
        real sin_out = std::hypot(in.x, in.y) * ior;
        if (sin_out >= 1) {
            // total reflection
            return 0.0f;
        }
        real cos_out = std::sqrt(1 - sin_out * sin_out);
        out_refract = Vector3(-ior * in.x, -ior * in.y, -cos_out * sgn(in.z));

        real rs, rp;

        rs = (cos_in - ior * cos_out) / (cos_in + ior * cos_out);
        rp = (ior * cos_in - cos_out) / (ior * cos_in + cos_out);

        rs = rs * rs;
        rp = rp * rp;
        return 1.0f - 0.5f * (rs + rp);
    }

    real get_ior(const Vector3 &in) const {
        return in.z < 0 ? inside_ior / outside_ior : outside_ior / inside_ior;
    }

    bool is_index_matched() const override {
        return inside_ior == outside_ior;
    }

    Vector3 sample_direction(const Vector3 &in, real u, real v, const Vector2 &uv) const {
        Vector3 out_reflect, out_refract;
        real p = get_refraction(in, out_reflect, out_refract);
        if (u < p) {
            return out_refract;
        }
        else {
            return out_reflect;
        }
    }

    virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        Vector3 out_reflect, out_refract;
        real p = get_refraction(in, out_reflect, out_refract);
        return in.z * out.z < 0 ? p : 1.0f - p;
    }

    virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        return Vector3(0.0f);
    }

    virtual bool is_delta() const override {
        return true;
    }

    virtual void sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir,
        Vector3 &f, real &pdf, SurfaceEvent &event, const Vector2 &uv) const override {
        out_dir = sample_direction(in_dir, u, v, uv);
        Vector3 out_reflect, out_refract;
        real p = get_refraction(in_dir, out_reflect, out_refract);
        if (u < p) {
            pdf = p;
            out_dir = out_refract;
        }
        else {
            pdf = 1 - p;
            out_dir = out_reflect;
        }
        auto color = color_sampler->sample3(uv);
        f = pdf * color * (1.0f / max(0.0f, std::abs(out_dir.z)));
        event = (int)SurfaceScatteringFlags::delta;
    }

    virtual real get_importance(const Vector2 &uv) const override {
        return luminance(color_sampler->sample3(uv));
    }
};

TC_IMPLEMENTATION(SurfaceMaterial, RefractiveMaterial, "refractive");

// TODO: Let's replace this with a true PBR material...
class PBRMaterial : public SurfaceMaterial {
protected:
    std::vector<std::shared_ptr<SurfaceMaterial> > materials;
    bool flag_is_delta;
public:
    virtual void initialize(const Config &config) override {
        SurfaceMaterial::initialize(config);
        auto diffuse_color_sampler = get_color_sampler(config, "diffuse");
        auto specular_color_sampler = get_color_sampler(config, "specular");
        auto glossy_color_sampler = get_color_sampler(config, "glossy");
        bool transparent = config.get("transparent", false);
        std::shared_ptr<SurfaceMaterial> diff_mat, glossy_mat;
        if (diffuse_color_sampler) {
            Config cfg;
            cfg.set("color_ptr", &diffuse_color_sampler);
            auto mat = std::make_shared<DiffuseMaterial>();
            mat->initialize(cfg);
            materials.push_back(mat);
        }
        if (transparent) {
            //TODO: load transparancy map...
            real ior = config.get_real("ior");
            Config cfg;
            cfg.set("ior", ior);
            auto mat = std::make_shared<RefractiveMaterial>();
            mat->initialize(cfg);
            materials.push_back(mat);
        }
        else if (specular_color_sampler) {
            Config cfg;
            cfg.set("color_ptr", &specular_color_sampler);
            glossy_mat = std::make_shared<ReflectiveMaterial>();
            glossy_mat->initialize(cfg);
            materials.push_back(glossy_mat);
        }
        if (glossy_color_sampler) {
            real glossiness = config.get_real("glossiness");
            Config cfg;
            cfg.set("color_ptr", &specular_color_sampler);
            cfg.set("glossiness", glossiness);
            glossy_mat = std::make_shared<GlossyMaterial>();
            glossy_mat->initialize(cfg);
            materials.push_back(glossy_mat);
        }
        flag_is_delta = false;
        for (auto &mat : materials) {
            if (!mat->is_delta()) {
                flag_is_delta = false;
            }
        }
    }

    DiscreteSampler get_material_sampler(const Vector2 uv) const {
        std::vector<real> luminances;
        for (auto &mat : materials) {
            real imp = mat->get_importance(uv);
            luminances.push_back(imp);
        }
        return DiscreteSampler(luminances, true);
    }

    void sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir, Vector3 &f, real &pdf, SurfaceEvent &event,
        const Vector2 &uv) const override {
        real mat_pdf, mat_cdf;
        int mat_id = get_material_sampler(uv).sample(u, mat_pdf, mat_cdf);
        if (mat_pdf == 0.0f) {
            f = Vector3(0.0f);
            pdf = 1.0f;
            event = (SurfaceEvent)SurfaceScatteringFlags::non_delta;
        }
        else {
            real rescaled_u = (u - (mat_cdf - mat_pdf)) / mat_pdf;
            assert(is_normal(rescaled_u));
            auto &mat = materials[mat_id];
            real submat_pdf;
            mat->sample(in_dir, rescaled_u, v, out_dir, f, submat_pdf, event, uv);
            if (SurfaceEventClassifier::is_delta(event)) {
                pdf = mat_pdf * submat_pdf;
            }
            else {
                pdf = probability_density(in_dir, out_dir, uv);
            }
        }
    }

    real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        real sum = 0;
        auto material_sampler = get_material_sampler(uv);
        for (int i = 0; i < (int)materials.size(); i++) {
            if (!materials[i]->is_delta())
                sum += materials[i]->probability_density(in, out, uv) *
                material_sampler.get_pdf(i);
        }
        return sum;
    }

    Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        Vector3 sum(0);
        for (auto &mat : materials) {
            if (!mat->is_delta())
                sum += mat->evaluate_bsdf(in, out, uv);
        }
        return sum;
    }

    bool is_delta() const override {
        return flag_is_delta;
    }
};

TC_IMPLEMENTATION(SurfaceMaterial, PBRMaterial, "pbr");

class PlainVolumeInterfaceMaterial : public SurfaceMaterial {
protected:
    virtual bool is_index_matched() const override {
        return true;
    }

    virtual void initialize(const Config &config) override {}

    virtual void sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir, Vector3 &f, real &pdf,
        SurfaceEvent &event, const Vector2 &uv) const override {
        out_dir = -in_dir;
        f = Vector3(1.0f) * abs(1.0f / in_dir.z);
        pdf = 1.0f;
        event = (int)SurfaceScatteringFlags::delta | (int)SurfaceScatteringFlags::index_matched;
        if (in_dir.z > 0) {
            event = event | (int)SurfaceScatteringFlags::entering;
        }
        else {
            event = event | (int)SurfaceScatteringFlags::leaving;
        }
    }

    virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        return 1.0f;
    };

    virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
        //return Vector3(1.0f) * abs(1.0f / out.z);
        return Vector3(0.0f);
    }

    virtual bool is_delta() const override {
        return true;
    }
};

TC_IMPLEMENTATION(SurfaceMaterial, PlainVolumeInterfaceMaterial, "plain_interface");

TC_NAMESPACE_END
