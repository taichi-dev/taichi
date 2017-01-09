/*
#include "spectrum_demo.h"

TC_NAMESPACE_BEGIN

void spectrum_demo(Config config) {
    int output_width = config.get_int("output_width");
    int output_height = config.get_int("output_height");
    float temperature_lower = config.get_float("temperature_lower");
    float temperature_upper = config.get_float("temperature_upper");
    bool tone_mapping = config.get_bool("tone_mapping");
    auto window = std::make_shared<GLWindow>(Config::load("render_view.cfg"));
    auto tr = std::make_shared<TextureRenderer>(window, output_width, output_height);
    auto buffer = Array2D<Vector3>(output_width, output_height);
    for (int i = 0; i < output_height; i++) {
        float t = temperature_lower + (temperature_upper - temperature_lower) * (i + 0.5f) / output_width;
        Vector3d color = Spectrum::get_instance().sample(t);
        if (!tone_mapping)
            color = color / max_component(color);
        for (int k = 0; k < 3; k++) {
            color[k] = max(color[k], 0.0);
        }
        for (int j = 0; j < output_width; j++) {
            buffer[j][i] = pow(color, 2.2f);
        }
    }
    for (int frame = 0; ; frame++) {
        auto _ = window->create_rendering_guard();
        if (tone_mapping) {
            tr->set_texture(ColorPreservingToneMapper::apply(buffer));
        }
        else {
            tr->set_texture(buffer);
        }
        tr->render();
    }
}

TC_NAMESPACE_END

*/
