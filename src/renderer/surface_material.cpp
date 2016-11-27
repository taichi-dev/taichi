#include "physics/physics_constants.h"
#include "surface_material.h"
#include "discrete_sampler.h"
#include "common/config.h"
#include "io/importer.h"

TC_NAMESPACE_BEGIN
    TC_INTERFACE_DEF(SurfaceMaterial, "material");

    class EmissiveMaterial : public SurfaceMaterial {
    public:
        virtual void initialize(const Config &config) override {
            SurfaceMaterial::initialize(config);
            set_color(config.get_vec3("color"));
        }

        virtual bool is_emissive() const override {
            return true;
        }

        virtual Vector3 sample_direction(const Vector3 &in, real u, real v, const Vector2 &uv) const override {
            return random_diffuse(Vector3(0, 0, in.z > 0 ? 1 : -1), u, v);
        };

        virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            if (in.z * out.z < eps) {
                return 0;
            }
            return out.z / pi;
        };

        virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            auto color = color_sampler->sample(uv);
            return (in.z * out.z > 0 ? 1.0f : 0.0f) * color; // No division by pi here.
        };
    };

    class DiffusiveMaterial : public SurfaceMaterial {
    public:
		void initialize(const Config &config) override {
			color_sampler = get_color_sampler(config, "diffuse");
		}

        virtual Vector3 sample_direction(const Vector3 &in, real u, real v, const Vector2 &uv) const override {
            Vector3 normal(0, 0, sgn(in.z));
            if (abs(in.z) > 1 - eps) {
                return random_diffuse(normal, u, v);
            } else {
                // We do the following other than the above to ensure correlation for MCMC...
                if (u > v) {
                    swap(u, v);
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
        };

        virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            if (in.z * out.z < eps) {
                return 0;
            }
            return out.z / pi;
        };

        virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            auto color = color_sampler->sample(uv);
            return (in.z * out.z > eps ? 1.0f : 0.0f) * color * (1.0f / pi);
        };
        virtual void
        sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir, Vector3 &f, real &pdf,
			SurfaceEvent &event, const Vector2 &uv) const override {
            out_dir = sample_direction(in_dir, u, v, uv);
            f = evaluate_bsdf(in_dir, out_dir, uv);
			event = (int)SurfaceScatteringFlags::non_delta;
			pdf = out_dir.z / pi;
        }
    };

    class GlossyMaterial : public SurfaceMaterial {
    protected:
        real glossiness = 300.0f;
    public:

        void set_glossiness(real glossiness) {
            this->glossiness = glossiness;
        }

        virtual Vector3 sample_direction(const Vector3 &in, real u, real v, const Vector2 &uv) const override {
            const Vector3 r = reflect(in);
            const Vector3 p = r.z > 1 - 1e-5f ? Vector3(0, 1, 0) : normalized(cross(Vector3(0, 0, 1), r));
            const Vector3 q = normalized(cross(r, p));
            const real phi = 2.0f * pi * u, d = pow(v, 1.0f / (glossiness + 1.0f));
            const real s = sin(phi), c = cos(phi), t = sqrt(1.0f - d * d);
            return (s * p + c * q) * t + d * r;
        };

        virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            if (in.z * out.z < eps) {
                return 0;
            }
            const Vector3 r = reflect(in);
            return (glossiness + 1.0f) / (2.0f * pi) * pow(max(dot(r, out), 0.0f), glossiness);
        };

        virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            if (in.z * out.z < eps) {
                return Vector3(0);
            }
            const Vector3 r = reflect(in);
            real t = min(max(dot(r, out), 0.0f), 1.0f);
            auto color = color_sampler->sample(uv);
            return color * (glossiness + 2.0f) / (2.0f * pi) * pow(t, glossiness)
                   / max(max(fabs(in.z), fabs(out.z)), 1e-7f);
        };
    };

    class ReflectiveMaterial : public SurfaceMaterial {
    public:

        virtual Vector3 sample_direction(const Vector3 &in, real u, real v, const Vector2 &uv) const override {
            return reflect(in);
        };

        virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            return 1;
        };

        virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            auto color = color_sampler->sample(uv);
            return color * (1.0f / out.z);
        };

        virtual bool is_delta() const override {
            return true;
        }
    };

    class SmallptRefractiveMaterial : public SurfaceMaterial {
		// from smallpt.cpp
		// the problem: when inside_ior == outside_ior, reflection still happens
    protected:
        real inside_ior = 1.5f;
        real outside_ior = 1.0f;
    public:

        void set_ior(real ior) {
            this->inside_ior = ior;
        }

        real get_refraction(const Vector3 &in, Vector3 &out_reflect, Vector3 &out_refract) const {
            // returns refraction probability
            out_reflect = reflect(in);
            bool into = in.z > 0;
            real ior = get_ior(in);
            real sin_out = hypot(in.x, in.y) * ior;
            if (sin_out >= 1) {
                // total reflection
                return 0.0f;
            }
            real cos_out = sqrt(1 - sin_out * sin_out);
            out_refract = Vector3(-ior * in.x, -ior * in.y, -cos_out * sgn(in.z));
            real a = inside_ior - outside_ior, b = inside_ior + outside_ior;
            real r0 = a * a / (b * b), c = 1 - (into ? in.z : out_refract.z);
            real reflectance = r0 + (1 - r0) * c * c * c * c * c;
            return 1.0f - reflectance;
        }

        real get_ior(const Vector3 &in) const {
            return in.z < 0 ? inside_ior / outside_ior : outside_ior / inside_ior;
        }

        virtual Vector3 sample_direction(const Vector3 &in, real u, real v, const Vector2 &uv) const override {
            Vector3 out_reflect, out_refract;
            real p = get_refraction(in, out_reflect, out_refract);
            if (u < p) {
                return out_refract;
            } else {
                return out_reflect;
            }
        }

        virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            Vector3 out_reflect, out_refract;
            real p = get_refraction(in, out_reflect, out_refract);
            return in.z * out.z < 0 ? p : 1.0f - p;
        }

        virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            Vector3 out_reflect, out_refract;
            real p = get_refraction(in, out_reflect, out_refract);
            real factor = in.z * out.z < 0 ? p : 1.0f - p;
            auto color = color_sampler->sample(uv);
            return factor * color * (1.0f / max(eps, std::abs(out.z)));
        }

        virtual bool is_delta() const override {
            return true;
        }
    };

    class RefractiveMaterial : public SurfaceMaterial {
		// See: https://en.wikipedia.org/wiki/Fresnel_equations
    protected:
        real inside_ior = 1.5f;
        real outside_ior = 1.0f;
    public:

        void set_ior(real ior) {
            this->inside_ior = ior;
        }

        real get_refraction(const Vector3 &in, Vector3 &out_reflect, Vector3 &out_refract) const {
            // returns refraction probability
            out_reflect = reflect(in);
            bool into = in.z > 0;
            real ior = get_ior(in);
			real cos_in = abs(in.z);
            real sin_out = hypot(in.x, in.y) * ior;
            if (sin_out >= 1) {
                // total reflection
                return 0.0f;
            }
            real cos_out = sqrt(1 - sin_out * sin_out);
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

        virtual Vector3 sample_direction(const Vector3 &in, real u, real v, const Vector2 &uv) const override {
            Vector3 out_reflect, out_refract;
            real p = get_refraction(in, out_reflect, out_refract);
            if (u < p) {
                return out_refract;
            } else {
                return out_reflect;
            }
        }

        virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            Vector3 out_reflect, out_refract;
            real p = get_refraction(in, out_reflect, out_refract);
            return in.z * out.z < 0 ? p : 1.0f - p;
        }

        virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            Vector3 out_reflect, out_refract;
            real p = get_refraction(in, out_reflect, out_refract);
            real factor = in.z * out.z < 0 ? p : 1.0f - p;
            auto color = color_sampler->sample(uv);
            return factor * color * (1.0f / max(eps, std::abs(out.z)));
        }

        virtual bool is_delta() const override {
            return true;
        }
    };

    class PBRMaterial : public SurfaceMaterial {
    protected:
        vector<std::shared_ptr<SurfaceMaterial> > materials;
        DiscreteSampler material_sampler;
        bool flag_is_delta;
    public:
        virtual void initialize(ptree &pt) override {
            Vector3 diffuse_color = load_vector3(pt.get("diffuse_color", "(0,0,0)"));
            Vector3 specular_color = load_vector3(pt.get("specular_color", "(0,0,0)"));
            real glossiness = pt.get("glossiness", 100.0f);
            bool transparent = pt.get("transparent", false);
            std::vector<real> luminances;
            std::shared_ptr<SurfaceMaterial> diff_mat, glossy_mat;
            if (luminance(diffuse_color) > 0) {
                diff_mat = std::make_shared<DiffusiveMaterial>();
                diff_mat->set_color(diffuse_color);
                materials.push_back(diff_mat);
                luminances.push_back(luminance(diffuse_color));
            }
            if (transparent) {
                // load transparancy...
                real ior = pt.get("ior", 1.5f);
                auto mat = std::make_shared<RefractiveMaterial>();
                mat->set_ior(ior);
                mat->set_color(Vector3(1, 1, 1));
                materials.push_back(mat);
                luminances.push_back(luminance(Vector3(1, 1, 1)));
            }
            if (luminance(specular_color) > 0) {
                if (glossiness > 0) { // glossy
                    glossy_mat = std::make_shared<GlossyMaterial>();
                    glossy_mat->set_color(specular_color);
                    static_cast<GlossyMaterial *>(glossy_mat.get())->set_glossiness(glossiness);
                    materials.push_back(glossy_mat);
                    luminances.push_back(luminance(specular_color));
                } else { // mirror
                    glossy_mat = std::make_shared<ReflectiveMaterial>();
                    glossy_mat->set_color(specular_color);
                    materials.push_back(glossy_mat);
                    luminances.push_back(luminance(specular_color));
                }
            }
                    foreach(ptree::value_type & v, pt.get_child("textures")) {
                            std::string filepath = v.second.get<std::string>("filepath");
                            real diff = v.second.get("use_map_color_diffuse", 0.0f);
                            real spec = v.second.get("use_map_color_spec", 0.0f);
                            auto tex = create_initialized_instance<Texture>("image", Config().set("filepath", filepath));
                            if (diff > 0 && diff_mat) {
                                diff_mat->set_color_sampler(tex);
                                P(filepath);
                            }
                            if (spec > 0 && glossy_mat) {
                                glossy_mat->set_color_sampler(tex);
                            }
                        }
            flag_is_delta = false;
            for (auto &mat : materials) {
                if (!mat->is_delta()) {
                    flag_is_delta = false;
                }
                assert_info(!mat->is_emissive(), "In PBR material, nothing can be emissive");
            }
            material_sampler.initialize(luminances);
        }

        virtual void initialize(const Config &config) override {
            SurfaceMaterial::initialize(config);
            Vector3 diffuse_color = config.get_vec3("diffuse_color");
            Vector3 specular_color = config.get_vec3("specular_color");
            real glossiness = config.get_real("glossiness");
            bool transparent = config.get_bool("transparent");
            std::vector<real> luminances;
            std::shared_ptr<SurfaceMaterial> diff_mat, glossy_mat;
            if (luminance(diffuse_color) > 0) {
                diff_mat = std::make_shared<DiffusiveMaterial>();
                diff_mat->set_color(diffuse_color);
                materials.push_back(diff_mat);
                luminances.push_back(luminance(diffuse_color));
            }
            if (transparent) {
                // load transparancy...
                real ior = config.get_real("ior");
                auto mat = std::make_shared<RefractiveMaterial>();
                mat->set_ior(ior);
                mat->set_color(Vector3(1, 1, 1));
                materials.push_back(mat);
                luminances.push_back(luminance(Vector3(1, 1, 1)));
            }
            if (luminance(specular_color) > 0) {
                if (glossiness > 0) { // glossy
                    glossy_mat = std::make_shared<GlossyMaterial>();
                    glossy_mat->set_color(specular_color);
                    static_cast<GlossyMaterial *>(glossy_mat.get())->set_glossiness(glossiness);
                    materials.push_back(glossy_mat);
                    luminances.push_back(luminance(specular_color));
                } else { // mirror
                    glossy_mat = std::make_shared<ReflectiveMaterial>();
                    glossy_mat->set_color(specular_color);
                    materials.push_back(glossy_mat);
                    luminances.push_back(luminance(specular_color));
                }
            }
            flag_is_delta = false;
            for (auto &mat : materials) {
                if (!mat->is_delta()) {
                    flag_is_delta = false;
                }
                assert_info(!mat->is_emissive(), "In PBR material, nothing can be emissive");
            }
            material_sampler.initialize(luminances);
        }

        void
        sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir, Vector3 &f, real &pdf, SurfaceEvent &event,
               const Vector2 &uv) const override {
            real mat_pdf, mat_cdf;
            int mat_id = material_sampler.sample(u, mat_pdf, mat_cdf);
            real rescaled_u = (u - (mat_cdf - mat_pdf)) / mat_pdf;
            assert(is_normal(rescaled_u));
            auto &mat = materials[mat_id];
            out_dir = mat->sample_direction(in_dir, rescaled_u, v, uv);
            f = mat->evaluate_bsdf(in_dir, out_dir, uv);
			event = 0;
			if(mat->is_delta())
				event |= (int)SurfaceScatteringFlags::delta;
            pdf = mat_pdf * mat->probability_density(in_dir, out_dir, uv);
        }

        real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            real sum = 0;
            for (int i = 0; i < (int) materials.size(); i++) {
                if (!materials[i]->is_delta())
                    sum += materials[i]->probability_density(in, out, uv) *
                           material_sampler.get_pdf(i);
            }
            return max(1e-7f, sum);
        }

        Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            Vector3 sum(0);
            for (auto &mat: materials) {
                if (!mat->is_delta())
                    sum += mat->evaluate_bsdf(in, out, uv);
            }
            return sum;
        }

        bool is_delta() const override {
            return flag_is_delta;
        }

    };

    TC_IMPLEMENTATION(SurfaceMaterial, DiffusiveMaterial, "diffusive");

    TC_IMPLEMENTATION(SurfaceMaterial, PBRMaterial, "pbr");

    TC_IMPLEMENTATION(SurfaceMaterial, EmissiveMaterial, "emissive");

    class PlainVolumeInterfaceMaterial : public SurfaceMaterial {
	protected:

		virtual bool is_index_matched() override {
			return true;
		}

        virtual void initialize(const Config &config) override {}

        virtual void sample(const Vector3 &in_dir, real u, real v, Vector3 &out_dir, Vector3 &f, real &pdf,
                            SurfaceEvent &event, const Vector2 &uv) const override {
			out_dir = -in_dir;
			f = Vector3(1.0f) * abs(1.0f / in_dir.z);
			pdf = 1.0f;
			event = (int)SurfaceScatteringFlags::delta | (int)SurfaceScatteringFlags::index_matched;
        }

        virtual real probability_density(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
            return 1.0f;
        };

        virtual Vector3 evaluate_bsdf(const Vector3 &in, const Vector3 &out, const Vector2 &uv) const override {
			return Vector3(1.0f) * abs(1.0f / out.z);
        }

        virtual bool is_delta() const override {
            return true;
        }
    };

	TC_IMPLEMENTATION(SurfaceMaterial, PlainVolumeInterfaceMaterial, "plain_interface");

TC_NAMESPACE_END
