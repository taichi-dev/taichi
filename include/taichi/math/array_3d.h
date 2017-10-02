/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/array_fwd.h>
#include "vector.h"
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <iterator>

TC_NAMESPACE_BEGIN

template <>
class IndexND<3> {
 private:
  int x[2], y[2], z[2];

 public:
  int i, j, k;
  Vector3 storage_offset;

  using Index3D = IndexND<3>;

  IndexND() {}

  IndexND(int x0,
          int x1,
          int y0,
          int y1,
          int z0,
          int z1,
          Vector3 storage_offset = Vector3(0.5f, 0.5f, 0.5f)) {
    x[0] = x0;
    x[1] = x1;
    y[0] = y0;
    y[1] = y1;
    z[0] = z0;
    z[1] = z1;
    i = x[0];
    j = y[0];
    k = z[0];
    this->storage_offset = storage_offset;
  }

  IndexND(Vector3i start,
          Vector3i end,
          Vector3 storage_offset = Vector3(0.5f, 0.5f, 0.5f)) {
    x[0] = start[0];
    x[1] = end[0];
    y[0] = start[1];
    y[1] = end[1];
    z[0] = start[2];
    z[1] = end[2];
    i = x[0];
    j = y[0];
    k = z[0];
    this->storage_offset = storage_offset;
  }

  IndexND(int i, int j, int k) {
    this->i = i;
    this->j = j;
    this->k = k;
  }

  IndexND(Vector3i p) {
    this->i = p[0];
    this->j = p[1];
    this->k = p[2];
  }

  void next() {
    k++;
    if (k == z[1]) {
      k = z[0];
      j++;
      if (j == y[1]) {
        j = y[0];
        i++;
        if (i == x[1]) {
        }
      }
    }
  }

  Index3D operator++() {
    this->next();
    return *this;
  }

  bool operator==(const Index3D &o) const {
    return (i == o.i && j == o.j && k == o.k);
  }

  bool operator<(const Index3D &o) const {
    return i < o.i || (i == o.i && (j < o.j)) ||
           (i == o.i && j == o.j && k < o.k);
  }

  bool operator!=(const Index3D &o) const {
    return !(i == o.i && j == o.j && k == o.k);
  }

  Index3D &to_end() {
    if (x[0] >= x[1] || y[0] >= y[1] || z[0] >= z[1])
      i = x[0];
    else
      i = x[1];
    j = y[0];
    k = z[0];
    return *this;
  }

  const Index3D &operator*() const { return *this; }

  Index3D &operator*() { return *this; }

  int operator[](int c) { return *(&i + c); }

  const int operator[](int c) const { return *(&i + c); }

  Index3D neighbour(int di, int dj, int dk) const {
    Index3D i = *this;
    i.i += di;
    i.j += dj;
    i.k += dk;
    return i;
  }

  Index3D neighbour(Vector3i d) const {
    Index3D i = *this;
    i.i += d.x;
    i.j += d.y;
    i.k += d.z;
    return i;
  }

  Index3D operator+(Vector3i d) const { return neighbour(d); }

  Vector3 get_pos() const {
    return Vector3((real)i + storage_offset.x, (real)j + storage_offset.y,
                   (real)k + storage_offset.z);
  }

  Vector3i get_ipos() const { return Vector3i(i, j, k); }
};

using Index3D = IndexND<3>;

template <>
class RegionND<3> {
 private:
  int x[2], y[2], z[2];
  Index3D index_begin;
  Index3D index_end;
  Vector3 storage_offset;

 public:
  using Region3D = RegionND<3>;

  RegionND() {}

  RegionND(int x0,
           int x1,
           int y0,
           int y1,
           int z0,
           int z1,
           Vector3 storage_offset = Vector3(0.5f, 0.5f, 0.5f)) {
    x[0] = x0;
    x[1] = x1;
    y[0] = y0;
    y[1] = y1;
    z[0] = z0;
    z[1] = z1;
    index_begin = Index3D(x0, x1, y0, y1, z0, z1, storage_offset);
    index_end = Index3D(x0, x1, y0, y1, z0, z1, storage_offset).to_end();
    this->storage_offset = storage_offset;
  }

  RegionND(Vector3i start,
           Vector3i end,
           Vector3 storage_offset = Vector3(0.5f, 0.5f, 0.5f)) {
    x[0] = start[0];
    x[1] = end[0];
    y[0] = start[1];
    y[1] = end[1];
    z[0] = start[2];
    z[1] = end[2];
    index_begin = Index3D(start, end, storage_offset);
    index_end = Index3D(start, end, storage_offset).to_end();
    this->storage_offset = storage_offset;
  }

  const Index3D begin() const { return index_begin; }

  Index3D begin() { return index_begin; }

  const Index3D end() const { return index_end; }

  Index3D end() { return index_end; }
};

using Region3D = RegionND<3>;

template <typename T>
class ArrayND<3, T> {
 protected:
  Region3D region;
  std::vector<T> data;
  typedef typename std::vector<T>::iterator iterator;
  int size;
  Vector3i res;
  int stride;
  Vector3 storage_offset =
      Vector3(0.5f, 0.5f, 0.5f);  // defualt : center storage
  struct Accessor2D {
    T *data;
    int offset;

    Accessor2D(T *data, int offset) : data(data), offset(offset) {}

    T *operator[](int i) const { return data + offset * i; }
  };

  struct ConstAccessor2D {
    const T *data;
    int offset;

    ConstAccessor2D(const T *data, int offset) : data(data), offset(offset) {}

    const T *operator[](int i) const { return data + offset * i; }
  };

 public:
  template <typename S>
  using Array3D = ArrayND<3, S>;

  int get_size() const { return size; }

  const Region3D &get_region() const { return region; }

  ArrayND(const Vector3i &resolution,
          T init = T(0),
          Vector3 storage_offset = Vector3(0.5f)) {
    initialize(resolution, init, storage_offset);
  }

  void initialize(const Vector3i &res,
                  T init = T(0),
                  Vector3 storage_offset = Vector3(0.5f)) {
    this->res = res;
    region = Region3D(0, res[0], 0, res[1], 0, res[2], storage_offset);
    size = res[0] * res[1] * res[2];
    stride = res[1] * res[2];
    data = std::vector<T>(size, init);
    this->storage_offset = storage_offset;
  }

  Array3D<T> same_shape(T init = T(0)) const {
    return Array3D<T>(res, init, storage_offset);
  }

  ArrayND(const Array3D<T> &arr) : ArrayND(arr.res) {
    this->data = arr.data;
    this->storage_offset = arr.storage_offset;
  }

  Array3D<T> operator+(const Array3D<T> &b) const {
    Array3D<T> o(res);
    assert(same_dim(b));
    for (int i = 0; i < size; i++) {
      o.data[i] = data[i] + b.data[i];
    }
    return o;
  }

  Array3D<T> operator-(const Array3D<T> &b) const {
    Array3D<T> o(res);
    assert(same_dim(b));
    for (int i = 0; i < size; i++) {
      o.data[i] = data[i] - b.data[i];
    }
    return o;
  }

  void operator+=(const Array3D<T> &b) {
    assert(same_dim(b));
    for (int i = 0; i < size; i++) {
      data[i] = data[i] + b.data[i];
    }
  }

  void operator-=(const Array3D<T> &b) {
    assert(same_dim(b));
    for (int i = 0; i < size; i++) {
      data[i] = data[i] - b.data[i];
    }
  }

  Array3D<T> &operator=(const Array3D<T> &arr) {
    this->res = arr.res;
    this->size = arr.size;
    this->stride = arr.stride;
    this->data = arr.data;
    this->region = arr.region;
    this->storage_offset = arr.storage_offset;
    return *this;
  }

  Array3D<T> &operator=(const T &a) {
    for (int i = 0; i < size; i++) {
      data[i] = a;
    }
    return *this;
  }

  ArrayND() {
    res = Vector3i(0);
    size = 0;
    stride = 0;
    data.resize(0);
  }

  ~ArrayND() {}

  void reset(T a) {
    for (int i = 0; i < size; i++) {
      data[i] = a;
    }
  }

  bool same_dim(const Array3D<T> &arr) const { return res == arr.res; }

  T dot(const Array3D<T> &b) const {
    T sum = 0;
    assert(same_dim(b));
    for (int i = 0; i < size; i++) {
      sum += this->data[i] * b.data[i];
    }
    return sum;
  }

  double dot_double(const Array3D<T> &b) const {
    double sum = 0;
    assert(same_dim(b));
    for (int i = 0; i < size; i++) {
      sum += this->data[i] * b.data[i];
    }
    return sum;
  }

  Array3D<T> add(T alpha, const Array3D<T> &b) const {
    Array3D<T> o(res);
    assert(same_dim(b));
    for (int i = 0; i < size; i++) {
      o.data[i] = data[i] + alpha * b.data[i];
    }
    return o;
  }

  void add_in_place(T alpha, const Array3D<T> &b) {
    for (int i = 0; i < size; i++) {
      data[i] += alpha * b.data[i];
    }
  }

  const Accessor2D operator[](int i) {
    return Accessor2D(&data[0] + i * stride, res[2]);
  }

  const ConstAccessor2D operator[](int i) const {
    return ConstAccessor2D(&data[0] + i * stride, res[2]);
  }

  const T &get(int i, int j, int k) const { return (*this)[i][j][k]; }

  const T &get(const Index3D &ind) const { return get(ind.i, ind.j, ind.k); }

  T &set_if_inside(const Index3D &ind, const T &val) {
    if (inside(ind))
      (*this)[ind] = val;
  }

  T &add_if_inside(const Index3D &ind, const T &val) {
    if (inside(ind))
      (*this)[ind] += val;
  }

  T get_copy(int i, int j, int k) const { return (*this)[i][j][k]; }

  void set(int i, int j, int k, const T &t) { (*this)[i][j][k] = t; }

  void set(const Index3D &ind, const T &t) { (*this)[ind] = t; }

  T abs_sum() const {
    T ret = 0;
    for (int i = 0; i < size; i++) {
      ret += abs(data[i]);
    }
    return ret;
  }

  T sum() const {
    T ret = 0;
    for (int i = 0; i < size; i++) {
      ret += data[i];
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
        printf("  [%d, %d, %d]\n", ind.i, ind.j, ind.k);
      }
    }
  }

  void print(std::string name = "", const char *temp = "%f") const {
    if (name.size())
      printf("%s[%dx%d]=", name.c_str(), res[0], res[1]);
    printf("\n");
    for (int k = 0; k < res[2]; k++) {
      for (int j = res[1] - 1; j >= 0; j--) {
        for (int i = 0; i < res[0]; i++) {
          printf(temp, (*this)[i][j][k]);
          printf(" ");
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }

  size_t get_data_size() const { return size * sizeof(T); }

  void set_pattern(int s) {
    for (int i = 0; i < size; i++) {
      data[i] = sinf(s * i + 231.0_f);
    }
  }

  bool inside(int i, int j, int k) const {
    return 0 <= i && i < res[0] && 0 <= j && j < res[1] && 0 <= k && k < res[2];
  }

  bool inside(const Vector3i &pos) const {
    return inside(pos[0], pos[1], pos[2]);
  }

  bool inside(const Index3D &index) const {
    return inside(index.i, index.j, index.k);
  }

  T sample(real x, real y, real z) const {
    x = clamp(x - storage_offset.x, 0.f, res[0] - 1.f - eps);
    y = clamp(y - storage_offset.y, 0.f, res[1] - 1.f - eps);
    z = clamp(z - storage_offset.z, 0.f, res[2] - 1.f - eps);
    int x_i = clamp(int(x), 0, res[0] - 2);
    int y_i = clamp(int(y), 0, res[1] - 2);
    int z_i = clamp(int(z), 0, res[2] - 2);
    real x_r = x - x_i;
    real y_r = y - y_i;
    real z_r = z - z_i;
    return lerp(
        z_r,
        lerp(x_r, lerp(y_r, get(x_i, y_i, z_i), get(x_i, y_i + 1, z_i)),
             lerp(y_r, get(x_i + 1, y_i, z_i), get(x_i + 1, y_i + 1, z_i))),
        lerp(x_r, lerp(y_r, get(x_i, y_i, z_i + 1), get(x_i, y_i + 1, z_i + 1)),
             lerp(y_r, get(x_i + 1, y_i, z_i + 1),
                  get(x_i + 1, y_i + 1, z_i + 1))));
  }

  T sample(const Vector3 &v) const { return sample(v.x, v.y, v.z); }

  T sample(const Index3D &v) const { return sample(v.get_pos()); }

  Vector3 get_storage_offset() const { return storage_offset; }

  T sample_relative_coord(const Vector3 &vec) const {
    real x = vec.x * res[0];
    real y = vec.y * res[1];
    real z = vec.z * res[2];
    return sample(x, y, z);
  }

  T sample_relative_coord(real x, real y, real z) const {
    x = x * res[0];
    y = y * res[1];
    z = z * res[2];
    return sample(x, y, z);
  }

  template <typename TT = T,
            typename std::enable_if_t<!std::is_class<TT>::value, int> = 0>
  T abs_max() const {
    T ret(0);
    for (int i = 0; i < size; i++) {
      ret = std::max(ret, abs(data[i]));
    }
    return ret;
  }

  template <typename TT = T, typename TS = typename TT::ScalarType>
  TS abs_max() const {
    TS ret(0);
    for (int i = 0; i < size; i++) {
      ret = std::max(ret, data[i].abs().max());
    }
    return ret;
  }

  auto begin() const { return data.cbegin(); }

  auto end() const { return data.cend(); }

  T &operator[](const Vector3i &pos) { return (*this)[pos.x][pos.y][pos.z]; }

  const T &operator[](const Vector3i &pos) const {
    return (*this)[pos.x][pos.y][pos.z];
  }

  T &operator[](const Index3D &index) {
    return (*this)[index.i][index.j][index.k];
  }

  const T &operator[](const Index3D &index) const {
    return (*this)[index.i][index.j][index.k];
  }

  Vector3i get_res() const { return res; }

  int get_width() const { return res[0]; }

  int get_height() const { return res[1]; }

  int get_depth() const { return res[2]; }

  bool empty() const { return size == 0; }

  T get_average() const {
    T sum(0);
    for (int i = 0; i < res[0]; i++) {
      for (int j = 0; j < res[1]; j++) {
        for (int k = 0; k < res[2]; k++) {
          sum += get(i, j, k);
        }
      }
    }
    return 1.0_f / size * sum;
  }

  bool inside(const Vector3 &pos, real tolerance = 1e-4f) const {
    return (-tolerance < pos.x && pos.x < res[0] + tolerance &&
            -tolerance < pos.y && pos.y < res[1] + tolerance &&
            -tolerance < pos.z && pos.z < res[2] + tolerance);
  }

  Region3D get_rasterization_region(Vector3 pos, int half_extent) const {
    int x = (int)floor(pos.x - storage_offset.x);
    int y = (int)floor(pos.y - storage_offset.y);
    int z = (int)floor(pos.z - storage_offset.z);
    return Region3D(
        std::max(0, x - half_extent + 1), std::min(res[0], x + half_extent + 1),
        std::max(0, y - half_extent + 1), std::min(res[1], y + half_extent + 1),
        std::max(0, z - half_extent + 1), std::min(res[2], z + half_extent + 1),
        storage_offset);
  }

  bool is_normal() const {
    for (auto v : (*this)) {
      if (!taichi::is_normal(v)) {
        return false;
      }
    }
    return true;
  }

  const std::vector<T> &get_data() const { return this->data; }

  std::vector<T> &get_data() { return this->data; }

  const int get_dim() const { return 3; }
};

template <typename T>
using Array3D = ArrayND<3, T>;

template <typename T>
void print(const Array3D<T> &arr) {
  arr.print("");
}

void test_array_3d();
TC_NAMESPACE_END
