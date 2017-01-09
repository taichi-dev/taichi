/*
#include "heat_transfer.h"
#include <taichi/visual/camera.h>
#include <taichi/visual/renderer.h>
#include "hdr/tone_mapper.h"
#include <taichi/visualization/texture_renderer.h>

TC_NAMESPACE_BEGIN

#define FOREACH_GRID for (int i = 0; i < grid_dim; i++) \
for (int j = 0; j < grid_dim; j++) \
for (int k = start[i][j]; k < grid_dim; k = next[i][j][k]) if (grid_inside_mesh(i, j, k))

    void HeatTransferSimulation::rasterize() {
        float tot = 0;
        FOREACH_GRID {
                        int tid = grid[i][j][k];
                        if (tid != -1 && grid_depth[i][j][k] < grid_cell_size) {
                            temperature[i][j][k] = scene->triangles[tid].temperature;
                        }
                        tot += temperature[i][j][k];
                    }
    }

    void HeatTransferSimulation::diffuse(real delta_t) {
        memcpy(temperature_tmp, temperature, sizeof(temperature));
        if (simulation_method == "forward") {
            FOREACH_GRID {
                            int dx[3]{1, 0, 0};
                            int dy[3]{0, 1, 0};
                            int dz[3]{0, 0, 1};
                            for (int d = 0; d < 3; d++) {
                                int ni = i + dx[d];
                                int nj = j + dy[d];
                                int nk = k + dz[d];
                                float c = -cond * delta_t;
                                if (grid_inside_mesh(ni, nj, nk)) {
                                    float heat = c * (temperature[i][j][k] - temperature[ni][nj][nk]);
                                    temperature_tmp[ni][nj][nk] -= heat;
                                    temperature_tmp[i][j][k] += heat;
                                }
                            }
                        }
        } else if (simulation_method == "backward") {
            real change = 10;
            while (change > 1e-4f) {
                change = 0;
                FOREACH_GRID {
                                int dx[6]{1, 0, 0, -1, 0, 0};
                                int dy[6]{0, 1, 0, 0, -1, 0};
                                int dz[6]{0, 0, 1, 0, 0, -1};
                                float sum = 0;
                                float c = cond * delta_t;
                                for (int d = 0; d < 6; d++) {
                                    int ni = i + dx[d];
                                    int nj = j + dy[d];
                                    int nk = k + dz[d];
                                    if (grid_inside_mesh(ni, nj, nk)) {
                                        sum += temperature_tmp[ni][nj][nk];
                                    }
                                }
                                sum = sum * c + temperature[i][j][k];
                                real old_val = temperature_tmp[i][j][k];
                                temperature_tmp[i][j][k] = sum / (1.0f + 2 * c);
                                change += sqr(temperature_tmp[i][j][k] - old_val);
                            }
            }
        } else {
            assert(false);
        }
        FOREACH_GRID {
                        temperature_tmp[i][j][k] = max(temperature_tmp[i][j][k], 0.0f);
                    }
        memcpy(temperature, temperature_tmp, sizeof(temperature));
    }

    void HeatTransferSimulation::unrasterize() {
        map<int, float> total_temp, total_weight;
        FOREACH_GRID {
                        int tid = grid[i][j][k];
                        if (tid != -1 && grid_depth[i][j][k] < grid_cell_size) {
                            total_temp[tid] += temperature[i][j][k];
                            total_weight[tid] += 1.0f;
                        }
                    }
        for (auto &t : total_weight) {
            if (t.second > eps) {
                float temp = total_temp[t.first] / t.second;
                scene->triangles[t.first].temperature = temp;
            }
        }

    }

    void HeatTransferSimulation::transfer_photon(Photon p, long long index) {
        real albedo = 0.5f;
        int offset = 3;
        for (int depth = 0; depth < 20; depth++) {
            Ray ray(p.pos, p.dir, 0);
            IntersectionInfo info = sg->query(ray);
            if (!info.intersected) {
                break;
            }
            int tid = info.triangle_id;
            p.pos = info.pos;

            Triangle &tri = scene->triangles[tid];
            if (tri.reflective) {
                Vector3 out_dir = reflect(ray.dir, info.normal);
                p.dir = out_dir;
            } else if (tri.refractive) {
                Vector3 out_dir = glm::refract(ray.dir, info.normal, !info.front ? 1.5f : 0.7f);
                p.dir = out_dir;
            } else {

                if (sampler->sample(offset, index) < 1 - albedo) {
                    scene->recieve_photon(tid, p.energy);
                    break;
                } else {
                    p.dir = random_diffuse(info.normal,
                                           sampler->sample(offset + 1, index), sampler->sample(offset + 2, index));
                }
            }
            offset += 3;
        }
    }

    void heat_demo(Config config) {
        string input_path = config.get_string("input_file");
        string output_path = config.get_string("output_file");
        int output_width = config.get_int("output_width");
        int output_height = config.get_int("output_height");
        int frame_number = config.get_int("frame_number");
        int output_per_frame = config.get_int("output_per_frame");
        float sub_divide_limit = config.get_float("sub_divide_limit");

        ptree pt;
        read_xml(input_path, pt);
        assert_info(!pt.empty(), "Can not read scene file");

        Scene scene(pt, sub_divide_limit);

        double starting_time = Time::get_time();

        Config renderer_cfg = Config::load("temperature_renderer.cfg");
        renderer_cfg.set("width", output_width).set("height", output_width);
        std::shared_ptr<Renderer> renderer = create_instance<Renderer>(config.get("renderer", "pt"));
        auto window = std::make_shared<GLWindow>(Config::load("simulation_view.cfg"));
        auto tr = std::make_shared<TextureRenderer>(window, output_width, output_height);
        ImageBuffer<Vector3> tex;

        HeatTransferSimulation *simulation = nullptr;
        //new HeatTransferSimulation(&scene, Config::load("heat_transfer_simulation.cfg"));
        window->add_mouse_move_callback_float([&](float x, float y) -> void {
            Vector3 c = renderer->get_output().sample(x, y, false);
            P(c.x);
            P(tex.sample_as_tex(x, y, false));
        });


        for (int frame = 0; frame < frame_number; frame++) {
            simulation->step();
            auto _ = window->create_rendering_guard();
            renderer->render_stage();
            std::shared_ptr<ImageBuffer<Vector3>> buffer =
                    std::make_shared<ImageBuffer<Vector3>>(renderer->get_output());
            //auto tex = (LogLuminance::apply(AverageToOne::apply(PhysicallyBasedHeatToneMapper::apply(*buffer))));
            tex = PBRTToneMapper::apply(PhysicallyBasedHeatToneMapper::apply(*buffer));
            if (frame % output_per_frame == 0) {
                char output_fn[1000];
                sprintf(output_fn, "%s/%05d.png", output_path.c_str(), frame);
                tex.write(output_fn);
            }
            tr->set_texture(tex);
            tr->render();
        }
    }

TC_NAMESPACE_END

        */
