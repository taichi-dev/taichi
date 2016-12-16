#pragma once

#include <memory>
#include <taichi/math/linalg.h>
#include <taichi/math/array_2d.h>
#include <taichi/system/threading.h>
#include <stb_image.h>
#include <stb_image_write.h>

TC_NAMESPACE_BEGIN

template <typename T>
class ImageBuffer : public Array2D<T>
{
public:
	ImageBuffer(int width, int height, T t) : Array2D<T>(width, height, t) {}
	ImageBuffer(int width, int height) : Array2D<T>(width, height) {}
	ImageBuffer() {}
	ImageBuffer(std::string filename) {
		load(filename);
	}
	void load(std::string filename) {
		int channels;
		float *data = stbi_loadf(filename.c_str(), &this->width, &this->height, &channels, 0);
		if (data == nullptr) {
			error("Image file not found: " + filename);
		}
		assert(channels == 3);
		this->initialize(this->width, this->height);
		for (int i = 0; i < this->width; i++) {
			for (int j = 0; j < this->height; j++) {
				float *pixel = data + ((this->height - 1 - j) * this->width + i) * channels;
				Vector3 color = vec3(pixel[0], pixel[1], pixel[2]);
				(*this)[i][j] = color;
			}
		}
		stbi_image_free(data);
	}

	void set_pixel(float x, float y, const T &pixel) {
		x *= this->width;
		y *= this->height;
		x -= 0.5f;
		y -= 0.5f;
		int int_x = (int)x;
		int int_y = (int)y;
		if (int_x < 0 || int_x >= this->width || int_y < 0 || int_y >= this->height)
			return;
		this->operator[](int_x)[int_y] = pixel;
	}

	T sample(float x, float y, bool interp = true) {
		x *= this->width;
		y *= this->height;
		x -= 0.5f;
		y -= 0.5f;
		x = clamp(x, 0.0f, this->width - 1.0f);
		y = clamp(y, 0.0f, this->height - 1.0f);
		int ix = clamp(int(x), 0, this->width - 2);
		int iy = clamp(int(y), 0, this->height - 2);
		if (!interp) {
			x = real(ix);
			y = real(iy);
		}
		T x_0 = lerp(y - iy, (*this)[ix][iy], (*this)[ix][iy + 1]);
		T x_1 = lerp(y - iy, (*this)[ix + 1][iy], (*this)[ix + 1][iy + 1]);
		return lerp(x - ix, x_0, x_1);
	}


	void write(std::string filename);

	void write_text(std::string content, float size, int dx, int dy);

};

template<typename T>
inline void ImageBuffer<T>::write(std::string filename)
{
	int comp = 3;
	std::vector<unsigned char> data(this->width * this->height * comp);
	for (int i = 0; i < this->width; i++) {
		for (int j = 0; j < this->height; j++) {
			for (int k = 0; k < comp; k++) {
				data[j * this->width * comp + i * comp + k] =
						(unsigned char)(255.0f * clamp(this->data[i * this->height + (this->height - j - 1)][k], 0.0f, 1.0f));
			}
		}
	}
	int write_result = stbi_write_png(filename.c_str(), this->width, this->height, comp, &data[0], comp * this->width);
	// assert_info((bool)write_result, "Can not write image file");
}

template<typename T>
class ImageAccumulator {
public:
	std::vector<Spinlock> locks;
	ImageAccumulator() {}

	ImageAccumulator(int width, int height) : width(width), height(height),
		buffer(width, height), counter(width, height)
	{
		for (int i = 0; i < width * height; i++) {
			locks.push_back(Spinlock());
		}
	}

	ImageBuffer<T> get_averaged(T default_value = T(0)) {
		ImageBuffer<T> result(width, height);
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				if (counter[i][j] > 0) {
					real inv = (real)1 / counter[i][j];
					result[i][j] = inv * buffer[i][j];
				}
				else {
					result[i][j] = default_value;
				}
			}
		}
		return result;
	}

	void accumulate(int x, int y, T val) {
		int lock_id = x * height + y;
		locks[lock_id].lock();
		counter[x][y] ++;
		buffer[x][y] += val;
		locks[lock_id].unlock();
	}

	void accumulate(ImageAccumulator<T> &other) {
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				counter[i][j] += other.counter[i][j];
				buffer[i][j] += other.buffer[i][j];
			}
		}
	}


	int get_width() const {
		return width;
	}

	int get_height() const {
		return height;
	}

private:
	ImageBuffer<T> buffer;
	ImageBuffer<int> counter;
	int width, height;
};

real estimate_error(const ImageBuffer<Vector3> &error_image);
ImageBuffer<Vector3> calculate_error_image(std::vector<std::shared_ptr<ImageBuffer<Vector3>>> buffers);
ImageBuffer<Vector3> combine(std::vector<std::shared_ptr<ImageBuffer<Vector3>>> buffers);

TC_NAMESPACE_END

