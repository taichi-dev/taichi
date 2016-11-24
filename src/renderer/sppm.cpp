#include "renderer.h"
#include "sampler.h"
#include "bsdf.h"
#include "sppm.h"

TC_NAMESPACE_BEGIN

    void SPPMRenderer::initialize(const Config &config) {
        Renderer::initialize(config);
        alpha = config.get("alpha", 0.666666667f);
        this->sampler = create_instance<Sampler>(config.get("sampler", "prand"));
		this->russian_roulette = config.get("russian_roulette", true);
        initial_radius = config.get_float("initial_radius");
        stochastic = config.get("stochastic", true);
        image.initialize(width, height);
        photon_counter = 0;
        stages = 0;
        radius2.initialize(width, height, initial_radius * initial_radius);
        flux.initialize(width, height, Vector3(0.0f));
        num_photons.initialize(width, height, 0LL);
        image_direct_illum.initialize(width, height);
        num_photons_per_stage = config.get("num_photons_per_stage", width * height);
        eye_ray_stages = 0;
    }

    void SPPMRenderer::render_stage() {
        hash_grid.clear_cache();
        if (stochastic || eye_ray_stages == 0) {
            eye_ray_pass();
            eye_ray_stages += 1;
        }
        hash_grid.build_grid();
        for (int i = 0; i < num_photons_per_stage; i++) {
            auto state_sequence = RandomStateSequence(sampler, photon_counter);
            trace_photon(state_sequence);
            photon_counter += 1;
        }
        stages += 1;
        for (auto &ind : image.get_region()) {
            image[ind] = 1.0f / (pi * radius2[ind]) / photon_counter * flux[ind] + image_direct_illum[ind] * (1.0f / eye_ray_stages);
        }
    }

    void SPPMRenderer::trace_eye_path(Ray &ray, const Vector2i &pixel) {
        Vector3 importance = Vector3(1.0f);
        for (int depth = 0; depth + 1 <= max_path_length; depth ++) {
            IntersectionInfo info = sg->query(ray);
            if (!info.intersected)
                return;

            Triangle &tri = scene->triangles[info.triangle_id];
            BSDF bsdf(scene, &info);
            Vector3 in_dir = -ray.dir;
            if (bsdf.is_emissive()) {
                if (min_path_length <= depth + 1 && depth + 1 <= max_path_length) {
                    image_direct_illum[pixel.x][pixel.y] += importance * bsdf.evaluate(bsdf.get_geometry_normal(), in_dir);
                }
                return;
            }
            Vector3 out_dir;
            Vector3 f;
            real pdf;
            Material::ScatteringEvent event;
            bsdf.sample(in_dir, rand(), rand(), out_dir, f, pdf, event);
            if (Material::is_delta(event)) { // continue tracing on specular surfaces
                importance = importance * f * bsdf.cos_theta(out_dir) / pdf;
                ray = Ray(info.pos, out_dir, 0);
            } else {
                HitPoint hit_point;
                hit_point.importance = importance;
                hit_point.pos = info.pos;
                hit_point.normal = info.normal;
                hit_point.pixel = pixel;
                hit_point.eye_out_dir = -ray.dir;
                hit_point.id = (int) hit_points.size();
                hit_point.path_length = depth + 1;
                hit_points.push_back(hit_point);
                real radius = sqrt(radius2[pixel.x][pixel.y]);
                hash_grid.push_back_to_all_cells_in_range(info.pos, radius, hit_point.id);
                return;
            }
        }
    }

    bool SPPMRenderer::trace_photon(StateSequence &rand) { // returns visibility
        bool visible = false;
        real pdf;
        Triangle &tri = scene->sample_triangle_light_emission(rand(), pdf);
        auto light_bsdf = BSDF(scene, tri.id);
        Vector3 pos = tri.sample_point(rand(), rand()),
                dir = light_bsdf.sample_direction(light_bsdf.get_geometry_normal(), rand(), rand());
        // should be divided by (1 / (area * pi)), where pi is the total solid angle
        Vector3 flux = light_bsdf.evaluate(tri.normal, dir) * (1.0f / pdf) * tri.area * pi;
        Ray ray(pos + dir * 1e-4f, dir, 0); // TODO: ... 1e-4f
        for (int depth = 0; depth + 1 <= max_path_length; depth++) {
            IntersectionInfo info = sg->query(ray);
            if (!info.intersected)
                break;
            Triangle &tri = scene->triangles[info.triangle_id];
            BSDF bsdf(scene, &info);
            Vector3 in_dir = -ray.dir;
            Vector3 out_dir;
            Vector3 f;
            Material::ScatteringEvent event;
            real pdf;
            if (bsdf.is_emissive()) {
                break;
            }
            bsdf.sample(in_dir, rand(), rand(), out_dir, f, pdf, event);
            Vector3 color = f * bsdf.cos_theta(out_dir) / pdf;
            if (Material::is_delta(event)) {
                // No vertex merging for delta BSDF
            } else {
                // Vertex merging
                int *begin = hash_grid.begin(info.pos);
                int *end = hash_grid.end(info.pos);
                for (int *p_hp_id = begin; p_hp_id < end; p_hp_id++) {
                    HitPoint &hp = hit_points[*p_hp_id];
                    Vector3 v = (hp.pos - info.pos);
                    int path_length = hp.path_length + depth + 1;
                    real &hp_radius2 = radius2[hp.pixel.x][hp.pixel.y];
                    if (min_path_length <= path_length && path_length <= max_path_length &&
                        dot(hp.normal, info.normal) > eps && dot(v, v) <= hp_radius2) {
                        long long &hp_num_photons = num_photons[hp.pixel.x][hp.pixel.y];
                        real g = (hp_num_photons * alpha + alpha) / (hp_num_photons * alpha + 1.0f);
                        hp_radius2 *= g;
                        Vector3 contribution = hp.importance * flux * bsdf.evaluate(in_dir, hp.eye_out_dir);
                        this->flux[hp.pixel.x][hp.pixel.y] = (this->flux[hp.pixel.x][hp.pixel.y] + contribution) * g;
                        hp_num_photons++;
                        visible = true;
                    }
                }
            }
            ray = Ray(info.pos, out_dir);
			//Russian roulette
			if (russian_roulette) {
				real p = max_component(color);
				if (rand() < p) {
					flux = (1.0f / p) * flux;
				} else {
					break;
				}
			}
            flux *= color;
        }
        return visible;
    }

    void SPPMRenderer::eye_ray_pass() {
        hash_grid.initialize(initial_radius, width * height * 10 + 7); // TODO: hash cell size should be shrinking...
        hit_points.clear();
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                Vector2 offset(float(i) / width, float(j) / height);
                Vector2 size(1.0f / width, 1.0f / height);
                Ray ray = camera->sample(offset, size);
                trace_eye_path(ray, Vector2i(i, j));
            }
        }
    }

    TC_IMPLEMENTATION(Renderer, SPPMRenderer, "sppm");
TC_NAMESPACE_END
