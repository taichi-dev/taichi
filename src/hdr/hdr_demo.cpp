#include "hdr_demo.h"

TC_NAMESPACE_BEGIN

void hdr_demo(Config config) {
	int output_width = config.get_int("output_width");
	int output_height = config.get_int("output_height");
	bool tone_mapping = config.get_bool("tone_mapping");
	string filename = config.get_string("filename");

	auto window = std::make_shared<GLWindow>(Config::load("hdr_view.cfg"));
	auto tr = std::make_shared<TextureRenderer>(window, output_width, output_height);
	auto buffer = ImageBuffer<Vector3>(output_width, output_height);
	ImageBuffer<Vector3> texture(filename);
	auto tone_mapped = PBRTToneMapper::apply(texture);

	window->add_mouse_move_callback_float([&](float x, float y) -> void {
		P(texture.sample(x, y, false));
		P(tone_mapped.sample(x, y, false));
	});
	P(texture.get_average());

	for (int frame = 0; ; frame++) {
		auto _ = window->create_rendering_guard();
		if (tone_mapping) {
			tr->set_texture(tone_mapped);
		}
		else {
			tr->set_texture(texture);
		}
		tr->render();
	}
}

TC_NAMESPACE_END

