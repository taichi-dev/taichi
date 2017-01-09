#pragma once

#include <memory>
#include <taichi/math/linalg.h>
#include <taichi/math/array_2d.h>
#include <taichi/system/threading.h>
#include <stb_image.h>
#include <stb_image_write.h>

TC_NAMESPACE_BEGIN

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

TC_NAMESPACE_END

