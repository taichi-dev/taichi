#pragma once

#include <taichi/visual/renderer.h>
#include <taichi/visual/sampler.h>
#include <taichi/visual/bsdf.h>

#include "hash_grid.h"

TC_NAMESPACE_BEGIN

    struct HitPoint {
        Vector2i pixel;
        Vector3 normal;
        Vector3 pos;
        Vector3 importance;
        Vector3 eye_out_dir;
        int id;
        int path_length = 0;
    };

    class SPPMRenderer : public Renderer {
    public:
        virtual void initialize(const Config &config) override;

        virtual void render_stage() override;

        virtual ImageBuffer<Vector3> get_output() override {
            return image;
        }

        virtual void trace_eye_path(Ray &ray, const Vector2i &pixel = Vector2i(-1, -1));

        virtual bool trace_photon(StateSequence &rand, real contribution_scaling=1.0f); // Returns visibility

        virtual void eye_ray_pass();

    protected:
        real alpha;
        real initial_radius;
        int num_photons_per_stage;
        HashGrid hash_grid;
        std::vector<HitPoint> hit_points;
        ImageBuffer<Vector3> image;
        ImageBuffer<Vector3> image_direct_illum;
        int stages;
        std::shared_ptr<Sampler> sampler;
        Array2D<real> radius2;
        Array2D<Vector3> flux;
        Array2D<long long> num_photons;
        int64 photon_counter;
        bool stochastic_eye_ray;
		bool russian_roulette;
        bool shrinking_radius;
        int eye_ray_stages;
    };
TC_NAMESPACE_END

