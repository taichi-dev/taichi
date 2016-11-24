#include "ANN/ANN.h"
#include <memory>
#include <vector>

#include "math/array_2d.h"
#include "point_cloud.h"
#include "visualization/texture_renderer.h"
#include "system/opengl.h"
#include "visualization/image_buffer.h"
#include "system/timer.h"

TC_NAMESPACE_BEGIN

template <typename T>
class NNI {
protected:
	std::vector<Vector2> points;
	std::vector<T> values;
public:
	NNI() {}
	NNI(const std::vector<Vector2> &points, const std::vector<T> &values) {
		initialize(points, values);
	}
	void initialize(const std::vector<Vector2> &points, const std::vector<T> &values) {
		this->points = points;
		this->values = values;
	}
	void rasterize(ImageBuffer<T> &buffer) {
		int width = buffer.get_width();
		int height = buffer.get_height();
		Array2D<Vector2> voronoi(width, height);
		Array2D<float> alpha(width, height, 0);
		NearestNeighbour2D nn(points);

		for (auto ind : voronoi.get_region()) {
			float x = ind.i + 0.5f;
			float y = ind.j + 0.5f;
			int index;
			float dist;
			nn.query(Vector2(x, y), index, dist);
			dist = sqrt(dist);
			for (int rx = int(x - dist - 1); rx <= int(x + dist + 1); rx++) {
				for (int ry = int(y - dist - 1); ry <= int(y + dist + 1); ry++) {
					if (alpha.inside(rx, ry) && hypot(x - rx - 0.5f, y - ry - 0.5f) <= dist) {
						alpha[rx][ry] += 1;
						buffer[rx][ry] += values[index];
						assert_info(max_component(values[index]) <= 1.0001f, "overflow!");
					}
				}
			}
		}
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				buffer[i][j] *= 1.0f / alpha[i][j];
				assert_info(max_component(buffer[i][j]) <= 1.0001f, "overflow!");
			}
		}
	}
};

template <typename T>
class GPUNNI : public NNI<T> {
protected:
	GLuint program, vao, vertex_buffer;
	GLuint framebuffer;
	GLuint texture_in;
	GLuint texture_out;

public:
	GPUNNI(const std::vector<Vector2> &points, const std::vector<T> &values) {
		initialize(points, values);
	}
	void initialize(const std::vector<Vector2> &points, const std::vector<T> &values) {
		this->points = points;
		this->values = values;

		assert_info(typeid(T) == typeid(Vector3), "Vector3 support only");
		{
			auto _ = GLWindow::get_gpgpu_window()->create_context_guard();
			program = load_program("nni", "nni");
		}
		framebuffer = -1;
		vao = -1;
		vertex_buffer = -1;
		texture_in = -1;
		texture_out = -1;
	}
	~GPUNNI() {
		auto _ = GLWindow::get_gpgpu_window()->create_context_guard();
		glDeleteProgram(program);
	}

	void create_texture_and_fbo(int width, int height, const std::vector<float> &texture_buffer) {
		if (framebuffer != -1) {
			glDeleteFramebuffers(1, &framebuffer);
		}
		if (texture_out != -1) {
			glDeleteTextures(1, &texture_out);
		}
		if (texture_in != -1) {
			glDeleteTextures(1, &texture_in);
		}

		glGenTextures(1, &texture_in);
		CGL;
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture_in);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, &texture_buffer[0]);

		glGenTextures(1, &texture_out);
		glBindTexture(GL_TEXTURE_2D, texture_out);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

		glGenFramebuffers(1, &framebuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture_out, 0);

		GLenum draw_buffers[]{ GL_COLOR_ATTACHMENT0 };
		glDrawBuffers(1, draw_buffers);
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			error("Framebuffer Error!");
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		CGL;

	}

	void create_vao_and_vbo(std::vector<float> &vertex_buffer_data) {
		if (vao != -1) {
			glDeleteVertexArrays(1, &vao);
		}
		if (vertex_buffer != -1) {
			glDeleteBuffers(1, &vertex_buffer);
		}

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vertex_buffer);

		glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertex_buffer_data.size(), &vertex_buffer_data[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(
			0,
			2,					// size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			sizeof(float) * 2,  // stride
			(void*)0            // array buffer offset
			);
		glBindVertexArray(0);
	}

	void rasterize(ImageBuffer<T> &buffer) {
		auto _ = GLWindow::get_gpgpu_window()->create_context_guard();

		int width = buffer.get_width();
		int height = buffer.get_height();
		Array2D<Vector2> voronoi(width, height);
		Array2D<float> alpha(width, height, 0);
		NearestNeighbour2D nn(this->points);

		std::vector<float> texture_buffer;

		for (auto ind : voronoi.get_region()) {
			float x = ind.i + 0.5f;
			float y = ind.j + 0.5f;
			int index;
			float dist;
			nn.query(Vector2(x, y), index, dist);
			for (int i = 0; i < 3; i++)
				texture_buffer.push_back(this->values[index][i]);
			texture_buffer.push_back(sqrt(dist));
		}

		create_texture_and_fbo(width, height, texture_buffer);
		CGL;

		vector<float> vertex_buffer_data;

		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				float x, y;
				x = float(i + 0.5f) / width;
				y = float(j + 0.5f) / height;
				vertex_buffer_data.push_back(x);
				vertex_buffer_data.push_back(y);
			}
		}

		CGL;
		create_vao_and_vbo(vertex_buffer_data);
		// framebuffer = 0; //NOTE!!
		glViewport(0, 0, width, height);
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
		glUseProgram(program);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture_in);
		glDisable(GL_DEPTH_TEST);
		glBindVertexArray(vao);
		glUniform1i(glGetUniformLocation(program, "texture"), 0);
		glUniform1f(glGetUniformLocation(program, "width"), (GLfloat)width);
		glUniform1f(glGetUniformLocation(program, "height"), (GLfloat)height);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);
		glDrawArrays(GL_POINTS, 0, width * height);
		glBindVertexArray(0);

		ImageBuffer<Vector4> gpu_buffer(width, height);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, &gpu_buffer[0][0]);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		bool all_zero = true;
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				Vector4 c = gpu_buffer[i][j];
				buffer[i][j] = Vector3(c.x, c.y, c.z) * (1.0f / c.w);
				if (c.x != 0 || c.y != 0 || c.z != 0 || c.w != 0) {
					all_zero = false;
				}
			}
		}
		assert_info(!all_zero, "all zero!!!");
		CGL;
	}
};

void point_cloud_demo(Config config) {
	using namespace std;
	vector<Vector2> points;
	int num_points = config.get("num_points", 10);

	LOAD_CONFIG(output_width, 512);
	LOAD_CONFIG(output_height, 512);

	ImageBuffer<Vector3> image;
	LOAD_CONFIG(use_image, false);
	if (use_image) {
		image.load(config.get_string("image_input"));
		output_height = output_width * image.get_height() / image.get_width();
	}

	for (int i = 0; i < num_points; i++) {
		// Uniform sampling the rectangle for width != height
		float diag = sqrt(rand());
		float x, y;
		float r = rand();
		float width_ratio = (float)output_width / (output_width + output_height);
		if (r < width_ratio) {
			x = rand() * diag;
			y = diag;
		}
		else {
			x = diag;
			y = rand() * diag;
		}
		points.push_back(Vector2(x, y) * Vector2(output_width, output_height));
	}
	vector<Vector3> colors;
	if (use_image) {
		for (int i = 0; i < num_points; i++) {
			colors.push_back(image.sample(points[i].x / output_width, points[i].y / output_height));
		}
	}
	else {
		for (int i = 0; i < num_points; i++) {
			colors.push_back(Vector3(rand() * 0.1 + 0.8, points[i].x / output_width, points[i].y / output_height));
		}
	}

	auto window = std::make_shared<GLWindow>(Config().set("width", output_width).set("height", output_height));
	auto tr = std::make_shared<TextureRenderer>(window, output_width, output_height);
	auto buffer = ImageBuffer<Vector3>(output_width, output_height);

	NearestNeighbour2D nn(points);
	if (config.get("integrate", true)) {
		vector<Vector3> color_acc(num_points, Vector3(0));
		vector<int> count(num_points, 0);
		for (int i = 0; i < output_width; i++) {
			for (int j = 0; j < output_height; j++) {
				float x = i + 0.5f;
				float y = j + 0.5f;
				int index;
				float dist;
				nn.query(Vector2(x, y), index, dist);
				color_acc[index] += image.sample((float)i / image.get_width(), (float)j / image.get_height());
				count[index]++;
			}
		}
		for (int i = 0; i < num_points; i++) {
			colors[i] = color_acc[i] * (1.0f / count[i]);
		}
	}

	if (!config.get("nii_interp", false)) {
		Time::Timer _("Voronoi Rasterization");
		for (int i = 0; i < output_width; i++) {
			for (int j = 0; j < output_height; j++) {
				float x = i + 0.5f;
				float y = j + 0.5f;
				int index;
				float dist;
				nn.query(Vector2(x, y), index, dist);
				buffer[i][j] = colors[index];
			}
		}
	}
	else {
		Time::Timer _("NII Rasterization");
		if (config.get("interpolator", "cpu") == "cpu") {
			NNI<Vector3> nni(points, colors);
			nni.rasterize(buffer);
		}
		else {
			GPUNNI<Vector3> nni(points, colors);
			nni.rasterize(buffer);
		}
	}

	ImageBuffer<Vector3> other_buffer(output_width, output_height);
	if (config.get("disp_diff", false)) {
		if (config.get("interpolator", "cpu") != "cpu") {
			NNI<Vector3> nni(points, colors);
			nni.rasterize(other_buffer);
		}
		else {
			GPUNNI<Vector3> nni(points, colors);
			nni.rasterize(other_buffer);
		}
		float diff_max = 0.0f;
		for (int i = 0; i < output_width; i++) {
			for (int j = 0; j < output_height; j++) {
				diff_max = max(diff_max, max_component(abs(buffer[i][j] - other_buffer[i][j])));
				buffer[i][j] = 1.0f * (buffer[i][j] - other_buffer[i][j]) + Vector3(0.5f);
			}
		}
		P(diff_max);
	}

	int frame = 0;
	while (true) {
		frame++;
		auto _ = window->create_rendering_guard();
		if (frame % 2 == 0) {
			tr->set_texture(buffer);
		}
		else {
			tr->set_texture(buffer);
		}
		tr->render();
	}
}

TC_NAMESPACE_END

