/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include "math_util.h"
#include "linalg.h"
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <iterator>

TC_NAMESPACE_BEGIN

class Index2D {
private:
    int x[2], y[2];
public:
    int i, j;
    //int offset;
    int stride;
    Vector2 storage_offset;

    Index2D() {}

    Index2D(int x0, int x1, int y0, int y1, Vector2 storage_offset = Vector2(0.5f, 0.5f)) {
        x[0] = x0;
        x[1] = x1;
        y[0] = y0;
        y[1] = y1;
        i = x[0];
        j = y[0];
        //offset = 0;
        stride = y[1] - y[0];
        this->storage_offset = storage_offset;
    }

    Index2D(int i, int j) {
        this->i = i;
        this->j = j;
    }


    void next() {
        j++;
        //offset++;
        if (j == y[1]) {
            j = y[0];
            i++;
            if (i == x[1]) {
            }
        }
    }

    Index2D operator++() {
        this->next();
        return *this;
    }

    bool operator==(const Index2D &o) const {
        return (i == o.i && j == o.j);
    }

    bool operator!=(const Index2D &o) const {
        return !(i == o.i && j == o.j);
    }

    Index2D &to_end() {
        i = x[1];
        j = y[0];
        //offset = (x[1] - x[0]) * (y[1] - y[0]);
        return *this;
    }

    const Index2D &operator*() const { return *this; }

    Index2D &operator*() { return *this; }

    int operator[](int c) { return *(&i + c); }

    const int operator[](int c) const { return *(&i + c); }

    Index2D neighbour(int di, int dj) const {
        Index2D i = *this;
        i.i += di;
        i.j += dj;
        return i;
    }

    Index2D neighbour(Vector2i d) const {
        Index2D i = *this;
        i.i += d.x;
        i.j += d.y;
        return i;
    }

    Index2D operator+(Vector2i d) const {
        return neighbour(d);
    }

    Vector2 get_pos() const {
        return Vector2((real)i + storage_offset.x, (real)j + storage_offset.y);
    }
};

class Region2D {
private:
    int x[2], y[2];
    Index2D index_begin;
    Index2D index_end;
    Vector2 storage_offset;
public:
    Region2D() {}

    Region2D(int x0, int x1, int y0, int y1, Vector2 storage_offset = Vector2(0.5f, 0.5f)) {
        x[0] = x0;
        x[1] = x1;
        y[0] = y0;
        y[1] = y1;
        index_begin = Index2D(x0, x1, y0, y1, storage_offset);
        index_end = Index2D(x0, x1, y0, y1, storage_offset).to_end();
        this->storage_offset = storage_offset;
    }

    const Index2D begin() const {
        return index_begin;
    }

    Index2D begin() {
        return index_begin;
    }

    const Index2D end() const {
        return index_end;
    }

    Index2D end() {
        return index_end;
    }
};

template <typename T>
struct Array2D {
protected:
    Region2D region;
    std::vector<T> data;
    typedef typename std::vector<T>::iterator iterator;
    int size;
    int width, height;
    Vector2 storage_offset = Vector2(0.5f, 0.5f); // defualt : center storage
public:
    template <typename P>
    friend Array2D<T> operator*(const P &b, const Array2D<T> &a);

    int get_size() const {
        return size;
    }

    const Region2D &get_region() const {
        return region;
    }

    void initialize(Vector2i resolution, T init = T(0), Vector2 storage_offset = Vector2(0.5f, 0.5f)) {
        initialize(resolution.x, resolution.y, init, storage_offset);
    }

    void initialize(int width, int height, T init = T(0), Vector2 storage_offset = Vector2(0.5f, 0.5f)) {
        //assert_info(width >= 2, "dim must be at least 2");
        //assert_info(height >= 2, "dim must be at least 2");
        this->width = width;
        this->height = height;
        region = Region2D(0, width, 0, height, storage_offset);
        size = width * height;
        data = std::vector<T>(size, init);
        this->storage_offset = storage_offset;
    }

    Array2D<T> same_shape(T init) const {
        return Array2D<T>(width, height, init, storage_offset);
    }

    Array2D<T> same_shape() const {
        return Array2D<T>(width, height);
    }

    Array2D(int width, int height, T init = T(0), Vector2 storage_offset = Vector2(0.5f, 0.5f)) {
        initialize(width, height, init, storage_offset);
    }

    Array2D(const Vector2i &res, T init = T(0), Vector2 storage_offset = Vector2(0.5f, 0.5f)) {
        initialize(res.x, res.y, init, storage_offset);
    }

    Array2D(const Array2D<T> &arr) : Array2D(arr.width, arr.height) {
        this->data = arr.data;
        this->storage_offset = arr.storage_offset;
    }

    template <typename P>
    Array2D<T> operator*(const P &b) const {
        Array2D<T> o(width, height);
        for (int i = 0; i < size; i++) {
            o.data[i] = b * data[i];
        }
        return o;
    }

    template <typename P>
    Array2D<T> operator/(const P &b) const {
        b = T(1) / b;
        return b * (*this);
    }

    Array2D<T> operator+(const Array2D<T> &b) const {
        Array2D<T> o(width, height);
        assert(same_dim(b));
        for (int i = 0; i < size; i++) {
            o.data[i] = data[i] + b.data[i];
        }
        return o;
    }

    Array2D<T> operator-(const Array2D<T> &b) const {
        Array2D<T> o(width, height);
        assert(same_dim(b));
        for (int i = 0; i < size; i++) {
            o.data[i] = data[i] - b.data[i];
        }
        return o;
    }

    void operator+=(const Array2D<T> &b) {
        assert(same_dim(b));
        for (int i = 0; i < size; i++) {
            data[i] = data[i] + b.data[i];
        }
    }

    void operator-=(const Array2D<T> &b) {
        assert(same_dim(b));
        for (int i = 0; i < size; i++) {
            data[i] = data[i] - b.data[i];
        }
    }

    Array2D<T> &operator=(const Array2D<T> &arr) {
        this->width = arr.width;
        this->height = arr.height;
        this->size = arr.size;
        this->data = arr.data;
        this->region = arr.region;
        this->storage_offset = arr.storage_offset;
        return *this;
    }

    Array2D<T> &operator=(const T &a) {
        for (int i = 0; i < size; i++) {
            data[i] = a;
        }
        return *this;
    }


    Array2D() {
        width = 0;
        height = 0;
        size = 0;
        data.resize(0);
    }

    ~Array2D() {
    }

    void reset(T a) {
        for (int i = 0; i < size; i++) {
            data[i] = a;
        }
    }

    bool same_dim(const Array2D<T> &arr) const {
        return width == arr.width && height == arr.height;
    }

    T dot(const Array2D<T> &b) const {
        T sum = 0;
        assert(same_dim(b));
        for (int i = 0; i < size; i++) {
            sum += this->data[i] * b.data[i];
        }
        return sum;
    }

    double dot_double(const Array2D<T> &b) const {
        double sum = 0;
        assert(same_dim(b));
        for (int i = 0; i < size; i++) {
            sum += this->data[i] * b.data[i];
        }
        return sum;
    }


    Array2D<T> add(T alpha, const Array2D<T> &b) const {
        Array2D o(width, height);
        assert(same_dim(b));
        for (int i = 0; i < size; i++) {
            o.data[i] = data[i] + alpha * b.data[i];
        }
        return o;
    }

    void add_in_place(T alpha, const Array2D<T> &b) {
        for (int i = 0; i < size; i++) {
            data[i] += alpha * b.data[i];
        }
    }

    T *operator[](int i) {
        return &data[0] + i * height;
    }

    const T *operator[](int i) const {
        return &data[0] + i * height;
    }

    const T &get(int i, int j) const {
        return (*this)[i][j];
    }

    const T &get(const Index2D &ind) const {
        return get(ind.i, ind.j);
    }


    T get_copy(int i, int j) const {
        return (*this)[i][j];
    }

    void set(int i, int j, const T &t) {
        (*this)[i][j] = t;
    }

    void set(const Index2D &ind, const T &t) {
        (*this)[ind] = t;
    }

    T abs_sum() const {
        T ret = 0;
        for (int i = 0; i < size; i++) {
            ret += std::abs(data[i]);
        }
        return ret;
    }

    T abs_max() const {
        T ret(0);
        for (int i = 0; i < size; i++) {
            ret = std::max(ret, abs(data[i]));
        }
        return ret;
    }

    T min() const {
        T ret = std::numeric_limits<T>::max();
        for (int i = 0; i < size; i++) {
            ret = std::min(ret, data[i]);
        }
        return ret;
    }

    T max() const {
        T ret = std::numeric_limits<T>::min();
        for (int i = 0; i < size; i++) {
            ret = std::max(ret, data[i]);
        }
        return ret;
    }

    void print_abs_max_pos() const {
        T ret = abs_max();
        for (auto &ind : get_region()) {
            if (abs(this->operator[](ind)) == ret) {
                printf("  [%d, %d]\n", ind.i, ind.j);
            }
        }
    }

    void print(std::string name = "") const {
        if (name.size())
            printf("%s[%dx%d]=", name.c_str(), width, height);
        printf("\n");
        for (int j = height - 1; j >= 0; j--) {
            for (int i = 0; i < width; i++) {
                printf("%+1.3f ", this->operator[](i)[j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    size_t get_data_size() const {
        return size * sizeof(T);
    }

    void set_pattern(int s) {
        for (int i = 0; i < size; i++) {
            data[i] = sinf(s * i + 231.0f);
        }
    }

    bool inside(int i, int j) const {
        return 0 <= i && i < width && 0 <= j && j < height;
    }

    bool inside(Index2D index) const {
        return inside(index.i, index.j);
    }

    T sample(real x, real y) const {
        x = clamp(x - storage_offset.x, 0.f, width - 1.f - eps);
        y = clamp(y - storage_offset.y, 0.f, height - 1.f - eps);
        int x_i = clamp(int(x), 0, width - 2);
        int y_i = clamp(int(y), 0, height - 2);
        real x_r = x - x_i;
        real y_r = y - y_i;
        return lerp(x_r,
                    lerp(y_r, get(x_i, y_i), get(x_i, y_i + 1)),
                    lerp(y_r, get(x_i + 1, y_i), get(x_i + 1, y_i + 1))
        );
    }

    T sample(const Vector2 &v) const {
        return sample(v.x, v.y);
    }

    T sample(const Index2D &v) const {
        return sample(v.get_pos());
    }

    Vector2 get_storage_offset() const {
        return storage_offset;
    }

    T sample_relative_coord(real x, real y) const {
        x = x * width;
        y = y * height;
        return sample(x, y);
    }

    T sample_relative_coord(const Vector2 &vec) const {
        real x = vec.x * width;
        real y = vec.y * height;
        return sample(x, y);
    }

    auto begin() const {
        return data.cbegin();
    }

    auto end() const {
        return data.cend();
    }

    T &operator[](const Vector2i &pos) {
        return (*this)[pos.x][pos.y];
    }

    const T &operator[](const Vector2i &pos) const {
        return (*this)[pos.x][pos.y];
    }

    T &operator[](const Index2D &index) {
        return (*this)[index.i][index.j];
    }

    const T &operator[](const Index2D &index) const {
        return (*this)[index.i][index.j];
    }

    int get_width() const {
        return width;
    }

    int get_height() const {
        return height;
    }

    bool empty() const {
        return !(width > 0 && height > 0);
    }

    T get_average() const {
        T sum(0);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                sum += get(i, j);
            }
        }
        return 1.0f / width / height * sum;
    }

    bool inside(const Vector2 &pos, real tolerance = 1e-4f) const {
        return (-tolerance <= pos.x && pos.x <= width + tolerance && -tolerance <= pos.y && pos.y < height + tolerance);
    }

    Region2D get_rasterization_region(Vector2 pos, int half_extent) const {
        int x = (int)floor(pos.x - storage_offset.x);
        int y = (int)floor(pos.y - storage_offset.y);
        return Region2D(std::max(0, x - half_extent + 1), std::min(width, x + half_extent + 1),
                        std::max(0, y - half_extent + 1),
                        std::min(height, y + half_extent + 1), storage_offset);
    }

    bool is_normal() const {
        for (auto v : (*this)) {
            if (!taichi::is_normal(v)) {
                return false;
            }
        }
        return true;
    }

    Array2D<T> rasterize(int width, int height) {
        Array2D<T> out(width, height);
        Vector2 actual_size;
        if (storage_offset == Vector2(0.0f, 0.0f)) {
            actual_size = Vector2(this->width - 1, this->height - 1);
        } else {
            actual_size = Vector2(this->width, this->height);
        }

        Vector2 scale_factor = actual_size / Vector2(width, height);

        for (auto &ind : Region2D(0, width, 0, height, Vector2(0.5f, 0.5f))) {
            Vector2 p = scale_factor * ind.get_pos();
            out[ind] = sample(p);
        }
        return out;
    }

    Array2D<T> rasterize_scale(int width, int height, int scale) {
        Array2D<T> out(width, height);
        for (auto &ind : out.get_region()) {
            out[ind] = (*this)[ind.i / scale][ind.j / scale];
        }
        return out;
    }

    const std::vector<T> &get_data() const {
        return this->data;
    }

    const int get_dim() const {
        return 2;
    }

    void flip(int axis) {
        if (axis == 0) {
            for (int i = 0; i < width / 2; i++) {
                for (int j = 0; j < height; j++) {
                    std::swap((*this)[i][j], (*this)[width - 1 - i][j]);
                }
            }
        } else {
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height / 2; j++) {
                    std::swap((*this)[i][j], (*this)[i][height - 1 - j]);
                }
            }
        }
    }

    // TODO: finally we are going to need a binary serializer

    void write_to_disk(const std::string &fn) {
        FILE *f = fopen(fn.c_str(), "wb");
        fwrite(&width, sizeof(width), 1, f);
        fwrite(&height, sizeof(height), 1, f);
        fwrite(&storage_offset, sizeof(storage_offset), 1, f);
        fwrite(&region, sizeof(region), 1, f);
        fwrite(&data[0], sizeof(data[0]), size, f);
        fclose(f);
    }

    bool read_from_disk(const std::string &fn) {
        FILE *f = fopen(fn.c_str(), "rb");
        if (f == nullptr) {
            return false;
        }
        size_t ret;
        ret = fread(&width, sizeof(width), 1, f);
        if (ret != 1) {
            return false;
        }
        ret = fread(&height, sizeof(height), 1, f);
        if (ret != 1) {
            return false;
        }
        ret = fread(&storage_offset, sizeof(storage_offset), 1, f);
        if (ret != 1) {
            return false;
        }
        ret = fread(&region, sizeof(region), 1, f);
        if (ret != 1) {
            return false;
        }
        initialize(Vector2i(width, height), T(0), storage_offset);
        ret = fread(&data[0], sizeof(data[0]), size, f);
        if (ret != size) {
            return false;
        }
        fclose(f);
        return true;
    }

    Array2D(const std::string &filename) {
        load(filename);
    }

    void load(const std::string &filename);

    void set_pixel(real x, real y, const T &pixel) {
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

    T sample_as_texture(real x, real y, bool interp = true) {
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

    void write(const std::string &filename);

    void write_text(const std::string &font_fn, const std::string &content, real size, int dx, int dy);
};

template <typename T, typename P>
Array2D<T> operator*(const P &b, const Array2D<T> &a) {
    Array2D<T> o(a.width, a.height);
    for (int i = 0; i < a.size; i++) {
        o.data[i] = b * a.data[i];
    }
    return o;
}

typedef Array2D<real> Array;

template <typename T>
void print(const Array2D<T> &arr) {
    arr.print("");
}

TC_NAMESPACE_END

