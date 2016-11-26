#pragma once

#include "camera.h"
#include "scene_geometry.h"
#include "visualization/image_buffer.h"
#include "system/timer.h"
#include "common/config.h"
#include "renderer.h"
#include "sampler.h"
#include "bsdf.h"

TC_NAMESPACE_BEGIN

// Reference : http://www.ci.i.u-tokyo.ac.jp/~hachisuka/smallpssmlt.cpp
    class SurfaceMaterial;
    enum class SurfaceScatteringFlags;

    struct Vertex {
    public:
        Vertex(const IntersectionInfo &inter, const BSDF &bsdf) : bsdf(bsdf) {
            pos = inter.pos;
            normal = inter.normal;
            triangle_id = inter.triangle_id;
            front = inter.front;
        }
        Vertex() { }
        Vector3 in_dir, out_dir;
        SurfaceEvent event;
        BSDF bsdf;
        Vector3 f;
        real pdf;
        Vector3 pos, normal;
        int triangle_id;
        bool front;
        bool connected = false; // if it is connected to the next vertex
    };

    typedef std::vector<Vertex> Path;

    struct Contribution {
        float x, y;
        int path_length;
        Vector3 c;

        Contribution() { };

        Contribution(float x, float y, int path_length, Vector3 c) :
                x(x), y(y), path_length(path_length), c(c) { }
    };

    struct PathContribution {
        std::vector<Contribution> contributions;
        real scaling = 1.0f;
        double total_contribution = 0.0;

        void push_back(const Contribution &c) {
            contributions.push_back(c);
            // We assume that luminance is used for SC...
            total_contribution += luminance(c.c);
        }
        bool empty() {
            return contributions.empty();
        }
        double get_total_contribution() {
            return total_contribution;
        }
        void set_scaling(real scaling) {
            this->scaling = scaling;
        }
        real get_scaling() const {
            return scaling;
        }
    };


    class BidirectionalRenderer : public Renderer {
    protected:
        int max_eye_events;
        int max_light_events;
        int stage_frequency;

        ImageBuffer<Vector3> buffer;
        std::shared_ptr<Sampler> sampler;
        real luminance_clamping;
        long long sample_count = 0;
        std::string print_path_policy;
        real vm_pdf_constant;

    public:
        virtual void initialize(const Config &config) override;

        void trace(Path &path, Ray r, int depth, int max_depth, StateSequence &rand);

        Path trace_eye_path(StateSequence &rand);

        Path trace_light_path(StateSequence &rand);

        bool connectable(int num_eye_vertices, int num_light_vertices, const Vertex &eye_end, const Vertex &light_end);

        Vector3d path_throughput(const Path &path);

        double path_pdf(const Path &path, const int num_eye_vert_spec,
                                   const int num_light_vert_spec);

        double path_total_pdf(const Path &path, bool including_connection, int merging_factor);


        PathContribution connect(const Path &eye_path, const Path &light_path,
                                 const int num_eye_vert_spec = -1,
                                 const int num_light_vert_spec = -1, const int merging_factor=0);

        double mis_weight(const Path &path, const int num_eye_vert_spec, const int num_light_vert_spec,
                          bool including_connection, int merging_factor);

        double geometry_term(const Vertex &e0, const Vertex &e1);

        double direction_to_area(const Vertex &current, const Vertex &next);

        void write_path_contribution(const PathContribution &pc, const real scaling = 1.0f);

        ImageBuffer<Vector3> get_output() override {
            ImageBuffer<Vector3> output(width, height);
            float r = 1.0f / sample_count;
            for (auto &ind : output.get_region()) {
                output[ind] = buffer[ind] * r;
            }
            return output;
        }
    };

TC_NAMESPACE_END

