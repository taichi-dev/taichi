#include "fluid_demo.h"
#include "fluid.h"
#include "visualization/texture_renderer.h"
#include <iomanip>
#include <iomanip>
#include <ctime>

#include "math/array_2d.h"

TC_NAMESPACE_BEGIN

class SimulationApplication {
private:
	int simulation_width;
	int simulation_height;
	int render_width;
	int render_height;
	std::shared_ptr<Fluid> fluid;
	float current_time;
	float delta_t;
	float cfl;
	float simulation_time;
	int frame_number = 0;
	std::shared_ptr<GLWindow> window;
	std::shared_ptr<TextureRenderer> tr;
	bool press_to_continue;
	std::string simulator_name;
	int current_frame_id;
	std::string output_path;
	bool finished;
	ImageBuffer<Vector3> last_image;
public:
	SimulationApplication(Config &config) {
		finished = false;
		current_frame_id = 0;
		output_path = config.get("output_path", "");
		simulation_width = config.get("simulation_width", 32);
		simulation_height = config.get("simulation_height", 32);
		int render_resolution = config.get("render_resolution", 600);
		delta_t = config.get("delta_t", 0.1f);
		simulation_time = config.get("simulation_time", 1.0f);
		current_time = 0;

		render_width = render_resolution;
		render_height = render_resolution * simulation_height / simulation_width;
		window = std::make_shared<GLWindow>(
			Config::load("fluid_view.cfg")
			.set("width", int(render_width))
			.set("height", int(render_height)));
		tr = std::make_shared<TextureRenderer>(window, simulation_width, simulation_height);
		press_to_continue = config.get("press_to_continue", false);

		window->add_mouse_move_callback_float([&](float x, float y) -> void {
		});

		simulator_name = config.get_string("fluid_simulator");
		fluid = create_fluid(simulator_name);
		fluid->initialize(config);
	}
	void update() {
		cout << "Updating " << simulator_name << " (t = " << current_time << ", " << std::setprecision(4) << current_time / simulation_time * 100.0f << "%)" << endl;
		auto buffer = ImageBuffer<Vector3>(simulation_width, simulation_height);
		if (simulation_time == 0 || current_time < simulation_time) {
			{
				Time::FPSCounter::count("Frame");
				auto _ = window->create_rendering_guard();

				fluid->step(delta_t);

				ImageBuffer<Vector3> texture(render_width, render_height);
				fluid->show(texture);
				tr->set_texture(texture);
				last_image = texture;
				tr->render();
				current_time += delta_t;
			}
			if (press_to_continue) {
				window->wait_for_key();
			}
			current_frame_id++;
		}
		else {
			if (!finished)
				finish();
		}
	}

	ImageBuffer<Vector3> get_last_image() {
		return last_image;
	}

	void finish() {

		finished = true;
	}

	bool is_finished() {
		return finished;
	}

};

void clear_frame_directory(string output_path) {
	assert_info(!output_path.empty(), "Path is empty!");
	std::string path = output_path + "/frames/frame*.png";
	std::replace(path.begin(), path.end(), '/', '\\');
	string command = "del " + path;
	system(command.c_str());
}

void make_video(string output_path, int width, int height) {
	// This can be DANGEROUS!
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	std::stringstream time_str;
	time_str << put_time(&tm, "%Y-%m-%d-%H-%M-%S");

	std::string command = "ffmpeg -framerate 24 -i " + output_path + "//frames//frame%d.png -s:v " + std::to_string(width) + "x" + std::to_string(height) + " -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + output_path + "//smoke-" + time_str.str() + ".mp4";
	system(command.c_str());

	clear_frame_directory(output_path);
}

class ImagePatchMerger {

};

void fluid_demo(Config config) {
	std::vector<std::string> application_cfgs = config.get_string_arr("experiments");
	string output_path = config.get("output_path", "");
	if (!output_path.empty()) {
		clear_frame_directory(output_path);
	}
	std::vector<std::shared_ptr<SimulationApplication>> applications;
	for (auto cfg : application_cfgs) {
		applications.push_back(std::make_shared<SimulationApplication>(Config(config).append(cfg)));
	}
	bool all_finished = false;
	int current_frame_id = 0;
	int patch_per_row;
	int patch_per_column;
	if (applications.size() == 1) {
		patch_per_row = 1;
		patch_per_column = 1;
	}
	else if (applications.size() == 2) {
		patch_per_row = 2;
		patch_per_column = 1;
	}
	else if (applications.size() == 3) {
		patch_per_row = 3;
		patch_per_column = 1;
	}
	else if (applications.size() == 4) {
		patch_per_row = 2;
		patch_per_column = 2;
	}
	else {
		error("too many applications!");
	}
	int video_width = -1, video_height = -1;
	while (!all_finished) {
		all_finished = true;
		ImageBuffer<Vector3> texture;
		vector<ImageBuffer<Vector3>> textures;
		for (auto app : applications) {
			app->update();
			if (!app->is_finished()) {
				all_finished = false;
			}
			textures.push_back(app->get_last_image());
		}
		int patch_width = textures[0].get_width();
		int patch_height = textures[0].get_height();
		texture.initialize(patch_width * patch_per_row, patch_height * patch_per_column);
		video_width = patch_width * patch_per_row;
		video_height = patch_height * patch_per_column;
		for (int i = 0; i < patch_width * patch_per_row; i++) {
			for (int j = 0; j < patch_height * patch_per_column; j++) {
				int texture_id = patch_per_row * (patch_per_column - 1 - (j / patch_height)) + (i / patch_width);
				if (texture_id < (int)textures.size()) {
					texture[i][j] = textures[texture_id][i % patch_width][j % patch_height];
				}
			}
		}
		if (!output_path.empty())
			texture.write(output_path + "//frames//frame" + std::to_string(current_frame_id) + ".png");
		current_frame_id++;
	}
	if (!output_path.empty() && video_width > 0 && video_height > 0)
		make_video(output_path, video_width, video_height);
}

TC_NAMESPACE_END

