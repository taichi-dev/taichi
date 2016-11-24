#include "camera.h"
#include "scene_geometry.h"
#include "visualization/image_buffer.h"
#include "system/timer.h"
#include "common/config.h"
#include "sampler.h"
#include "bidirectional_renderer.h"
#include "hash_grid.h"

TC_NAMESPACE_BEGIN
    class VCMRenderer : public BidirectionalRenderer {
    protected:
        int num_stages;
        real initial_radius;
        HashGrid hash_grid;
        std::vector<Path> light_paths;
        std::vector<Path> light_paths_for_connection;
        real radius;
        int n_samples_per_stage;
        ImageBuffer<Vector3> bdpm_image;
        real alpha;
        bool use_vc;
        bool use_vm;

    public:
        virtual void initialize(const Config &config) override {
            BidirectionalRenderer::initialize(config);
            num_stages = 0;
            initial_radius = config.get_float("initial_radius");
            use_vc = config.get("use_vc", true);
            use_vm = config.get("use_vm", true);
            alpha = config.get("alpha", 0.66667f);
            bdpm_image.initialize(width, height, Vector3(0.0f, 0.0f, 0.0f));
            radius = initial_radius;
            n_samples_per_stage = width * height / stage_frequency;
        }

        PathContribution vertex_merge(const Path &full_eye_path) {
            PathContribution pc;
            real radius2 = radius * radius;
            for (int num_eye_vertices = 2; num_eye_vertices <= (int) full_eye_path.size(); num_eye_vertices++) {
                Path eye_path(full_eye_path.begin(), full_eye_path.begin() + num_eye_vertices);
                Vector3 merging_pos = eye_path.back().pos;
                int *begin = hash_grid.begin(merging_pos), *end = hash_grid.end(merging_pos);
                for (int *light_path_id_pointer = begin; light_path_id_pointer < end; light_path_id_pointer++) {
                    int light_path_id = *light_path_id_pointer;
                    Path light_path = light_paths[light_path_id];
                    int path_length = (int) eye_path.size() + (int) light_path.size() - 2;
                    int num_light_vertices = (int) light_path.size();
                    Vertex merging_vertex_eye = eye_path.back();
                    Vertex merging_vertex_light = light_path.back();
                    if (Material::is_delta(merging_vertex_eye.event) ||
                        Material::is_delta(merging_vertex_light.event)) {
                        // Do not connect Delta BSDF
                        continue;
                    }
                    Vector3 v = merging_vertex_eye.pos - merging_vertex_light.pos;
                    if (min_path_length <= path_length && path_length <= max_path_length &&
                        dot(merging_vertex_eye.normal, merging_vertex_light.normal) > eps && dot(v, v) <= radius2) {
                        // Screen coordinates
                        Vector3 camera_direction = normalize(eye_path[1].pos - eye_path[0].pos);
                        real screen_u, screen_v;
                        camera->get_pixel_coordinate(camera_direction, screen_u, screen_v);
                        if (!(0 <= screen_u && screen_u < 1 && 0 <= screen_v && screen_v < 1)) {
                            //assert_info(-eps <= screen_u && screen_u <= 1 + eps && -eps <= screen_v && screen_v <= 1 + eps,
                            //            "Eye ray outside camera ???");
                            // TODO: is this caused by non-zero radius?
                            continue;
                        }
                        screen_u = clamp(screen_u, 0.0f, 1.0f);
                        screen_v = clamp(screen_v, 0.0f, 1.0f);
                        eye_path.back().connected = true;
                        Path full_path;
                        full_path.resize(
                                num_eye_vertices + num_light_vertices - 1); // note that last light vertex is deleted
                        for (int i = 0; i < num_eye_vertices; i++) full_path[i] = eye_path[i];
                        for (int i = 0; i < num_light_vertices - 1; i++) full_path[path_length - i] = light_path[i];
                        // evaluate the path
                        Vector3 f = path_throughput(full_path);
                        if (max_component(f) <= 0.0f) {
                            //printf("f\n");
                            continue;
                        }
                        double p = path_pdf(full_path, num_eye_vertices, num_light_vertices);
                        if (p <= 0.0f) {
                            //printf("p\n");
                            continue;
                        }
                        double w = mis_weight(full_path, num_eye_vertices, num_light_vertices, use_vc, n_samples_per_stage);
                        if (w <= 0.0f) {
                            //printf("w\n");
                            continue;
                        }
                        Vector3 c = f * float(w / p);
                        if (max_component(c) <= 0.0) continue;
                        pc.push_back(Contribution(screen_u, screen_v, path_length, c));
                    }
                }
            }
            return pc;
        }

        virtual void render_stage() override {
            radius = initial_radius * pow(num_stages + 1, -(1.0f - alpha) / 2.0f);
            vm_pdf_constant = pi * radius * radius;
            hash_grid.initialize(radius, width * height * 10 + 7);
            light_paths.clear();
            light_paths_for_connection.clear();
            // Generate light paths (photons)
            for (int k = 0; k < n_samples_per_stage; k++) {
                auto state_sequence = RandomStateSequence(sampler, sample_count * 2 + k); // TODO: wrong...
                Path light_path = trace_light_path(state_sequence);
                light_paths_for_connection.push_back(light_path);
                if (use_vm) {
                    for (int num_light_vertices = 2; num_light_vertices <= (int) light_path.size(); num_light_vertices++) {
                        Path partial_light_path(light_path.begin(), light_path.begin() + num_light_vertices);
                        hash_grid.push_back_to_all_cells_in_range(partial_light_path.back().pos, radius, (int) light_paths.size());
                        light_paths.push_back(partial_light_path);
                    }
                }
            }
            hash_grid.build_grid();
            // Generate eye paths (importons)
            for (int k = 0; k < n_samples_per_stage; k++) {
                auto state_sequence = RandomStateSequence(sampler, sample_count * 2 + n_samples_per_stage + k);
                Path eye_path = trace_eye_path(state_sequence);
                if (use_vm) {
                    write_path_contribution(vertex_merge(eye_path));
                }
                if (use_vc) {
                    write_path_contribution(connect(eye_path, light_paths_for_connection[k], -1, -1, (int)use_vm * n_samples_per_stage));
                }
            }
            sample_count += n_samples_per_stage;
        }
    };

    TC_IMPLEMENTATION(Renderer, VCMRenderer, "vcm");

TC_NAMESPACE_END

