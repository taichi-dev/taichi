#pragma once

#include <vector>
#include "system/opengl.h"
#include "image_buffer.h"

TC_NAMESPACE_BEGIN

using glm::vec4;

class TextureRenderer {
private:
    static GLuint program, vbo;
	static bool shared_resources_initialized;
	GLuint vao;
	GLuint texture;
    int width, height;
    vector<unsigned char> image;
	std::shared_ptr<GLWindow> context;
public:
    TextureRenderer(std::shared_ptr<GLWindow> window, int height, int width);

	void resize(int height, int width);

    void reset();

    void set_pixel(int x, int y, vec4 color);

	template <typename T>
	void set_texture(ImageBuffer<T> image);

    void render();

    ~TextureRenderer();

	static vec4 to_vec4(float dat) {
		return vec4(dat);
	}
	static vec4 to_vec4(unsigned char dat) {
		return vec4(dat / 255.0f);
	}
	static vec4 to_vec4(vec2 dat) {
		return vec4(dat, 0, 1);
	}
	static vec4 to_vec4(vec3 dat) {
		return vec4(dat, 1);
	}   
	static vec4 to_vec4(vec4 dat) {
		return dat;
	}   
};

template<typename T>
inline void TextureRenderer::set_texture(ImageBuffer<T> image)
{
	// assert_info(image.get_width() == width && image.get_height() == height, "Texture size mismatch!");
	resize(image.get_width(), image.get_height());
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			set_pixel(i, j, to_vec4(image[i][j]));
		}
	}
}

TC_NAMESPACE_END

