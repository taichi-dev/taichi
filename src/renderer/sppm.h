#pragma once

#include "camera.h"
#include "scene_geometry.h"
#include "visualization/image_buffer.h"
#include "system/timer.h"
#include "common/config.h"
#include "sampler.h"
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

        virtual bool trace_photon(StateSequence &rand); // Returns visibility

        virtual void eye_ray_pass();

    protected:
        real alpha;
        real initial_radius;
        int num_photons_per_stage;
        HashGrid hash_grid;
        vector<HitPoint> hit_points;
        ImageBuffer<Vector3> image;
        ImageBuffer<Vector3> image_direct_illum;
        int stages;
        std::shared_ptr<Sampler> sampler;
        Array2D<real> radius2;
        Array2D<Vector3> flux;
        Array2D<long long> num_photons;
        long long photon_counter;
        bool stochastic;
        int eye_ray_stages;
		bool russian_roulette;
    };
TC_NAMESPACE_END

