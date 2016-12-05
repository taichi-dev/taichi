/*
#include "rendering.h"
#include "renderer.h"
#include "hdr/tone_mapper.h"
#include "visualization/texture_renderer.h"

TC_NAMESPACE_BEGIN
    void rendering_demo(Config config) {
        string input_path = config.get_string("input_file");
        real exposure = config.get("exposure", 1.0f);

        ptree pt;
        read_xml(input_path, pt);
        assert_info(!pt.empty(), "Can not read scene file");

        auto scene = std::make_shared<Scene>(pt, 0.0f);
        int output_width = scene->resolution_x;
        int output_height = scene->resolution_y;

        double starting_time = Time::get_time();
        config.append(config.get_string("renderer_cfg"));
        config.set("width", output_width);
        config.set("height", output_height);
        auto renderer = create_instance<Renderer>(config.get("renderer", "pt"));
        renderer->initialize(config);
        renderer->set_scene(scene);
        auto window = std::make_shared<GLWindow>(Config::load("render_view.cfg").set("title", config.get_string("renderer_cfg")));
        auto tr = std::make_shared<TextureRenderer>(window, output_width, output_height);
        string output_path = config.get_string("output_path");

        int stages = 0;

        for (int frame = 0; ; frame++) {
            double time_elapsed = Time::get_time() - starting_time;
            renderer->render_stage();
            stages++;
            real time_per_stage = (real) (time_elapsed) / stages;
            P(time_per_stage);
            auto tex = renderer->get_output();
            Vector3 sum(0);
            for (int i = 0; i < tex.get_width(); i++) {
                for (int j = 0; j < tex.get_height(); j++) {
                    sum += tex[i][j];
                }
            }
            Vector3 pixel_average = 1.0f / output_width / output_height * sum;
            P(pixel_average);
            auto tex2 = AverageToOne::apply(tex);
            for (auto &ind : tex2.get_region()) {
                tex2[ind] *= exposure;
            }
            auto to_disp = GammaCorrection::apply(tex2);
            tr->set_texture(to_disp);
            auto _ = window->create_rendering_guard();
            tr->render();
            tex2.write(output_path);
        }
    }

void rendering_demo(Config config) {
    string input_path = config.get_string("input_file");
    int num_instances = config.get_int("num_instances");
    double time_limit = config.get_double("time_limit");
    int stage_limit = config.get_int("stage_limit");
    real exposure = config.get("exposure", 1.0f);

    ptree pt;
    read_xml(input_path, pt);
    assert_info(!pt.empty(), "Can not read scene file");

    Scene scene(pt, 0.0f);
    int output_width = scene.resolution_x;
    int output_height = scene.resolution_y;

    struct NoiseInfo {
        double noise;
        double time;
    };

    vector<NoiseInfo> noise_info;

    int current_instance = 0;
    double starting_time = Time::get_time();
    ImageBuffer<Vector3> error_image(output_width, output_height);
    real e_acc = -1;

    config.append(config.get_string("renderer_cfg"));
    config.set("width", output_width);
    config.set("height", output_height);
    std::vector<std::shared_ptr<Renderer>> renderers(num_instances);
    bool enable_test = config.get_bool("enable_test") && num_instances > 1;
    for (int i = 0; i < num_instances; i++) {
        renderers[i] = std::shared_ptr<Renderer>(create_instance<Renderer>(config.get("renderer", "pt")));
        renderers[i]->initialize(&scene, config);
    }
    auto window1 = std::make_shared<GLWindow>(Config::load("render_view.cfg").set("title", config.get_string("renderer_cfg")));
    auto window2 = enable_test ? std::make_shared<GLWindow>(Config::load("error_view.cfg")) : nullptr;
    auto tr1 = std::make_shared<TextureRenderer>(window1, output_width, output_height);
    auto tr2 = enable_test ? std::make_shared<TextureRenderer>(window2, output_width, output_height) : nullptr;
    string output_path = config.get_string("output_path");

    int stages = 0;
    ImageBuffer<Vector3> tex;

    window1->add_mouse_move_callback_float([&](float x, float y) -> void {
        if (!tex.empty()) P(tex.sample(x, y, false));
    });

    for (int frame = 0;; frame++) {
        double time_elapsed = Time::get_time() - starting_time;
        if ((time_limit == 0 || time_elapsed < time_limit) && (stage_limit == 0 || stages < stage_limit)) {
            {
                Time::Timer _("Render Stage");
                renderers[current_instance]->render_stage();
            }
            stages++;
        } else {
        }
        real time_per_stage = (real) (time_elapsed) / stages;
        P(time_per_stage);
        current_instance = (current_instance + 1) % num_instances;
        if (current_instance == 0) {
            auto _ = window1->create_rendering_guard();
            std::vector<std::shared_ptr<ImageBuffer<Vector3>>> buffers(num_instances);
            for (int i = 0; i < num_instances; i++) {
                buffers[i] = std::shared_ptr<ImageBuffer<Vector3>>(
                        new ImageBuffer<Vector3>(renderers[i]->get_output()));
            }
            if (enable_test) {
                NoiseInfo info;
                error_image = calculate_error_image(buffers);
                info.noise = estimate_error(error_image);
                info.time = Time::get_time() - starting_time;

                if (noise_info.size() > 5) {
                    NoiseInfo last = noise_info.back();
                    real e = (real) ((log(info.noise) - log(last.noise)) / (log(info.time) - log(last.time)));
                    if (e_acc == -1) {
                        e_acc = e;
                    } else {
                        float alpha = 0.95f;
                        e_acc = e_acc * alpha + (1 - alpha) * e;
                    }
                    P(e_acc);
                    P(log(info.noise));
                    real estimated_log_error_in_one_error = real(
                            log(noise_info.front().noise) + e_acc * log(3600.0f));
                    P(estimated_log_error_in_one_error);
                }
                noise_info.push_back(info);
            }

            tex = combine(buffers);
            Vector3 sum(0);
            for (int i = 0; i < tex.get_width(); i++) {
                for (int j = 0; j < tex.get_height(); j++) {
                    sum += tex[i][j];
                }
            }
            Vector3 pixel_average = 1.0f / output_width / output_height * sum;
            P(pixel_average);
            //auto to_disp = PBRTToneMapper::apply(tex);
            auto tex2 = AverageToOne::apply(tex);
            for (auto &ind : tex2.get_region()) {
                tex2[ind] *= exposure;
            }
            auto to_disp = GammaCorrection::apply(tex2);
            tr1->set_texture(to_disp);
            if (stages % 10 == -1) {
                to_disp.write(output_path);
                FILE *f = fopen("C:/tmp/hist.bin", "wb");
                vector<real> lum;
                for (int i = 0; i < output_width; i++) {
                    for (int j = 0; j < output_height; j++) {
                        lum.push_back(luminance(tex[i][j]));
                    }
                }
                fwrite(&lum[0], sizeof(float), output_width * output_height, f);
                fclose(f);
            }
            tr1->render();
        }
        if (enable_test) {
            auto _ = window2->create_rendering_guard();
            tr2->set_texture(PBRTToneMapper::apply(error_image));
            tr2->render();
        }
    }
}

TC_NAMESPACE_END
 */

