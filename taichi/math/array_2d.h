/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <iterator>

#include "array_fwd.h"
#include "linalg.h"

TI_NAMESPACE_BEGIN

template <>
class IndexND<2> {
 private:
  int x[2], y[2];

 public:
  using Index = IndexND<2>;

  int i, j;
  // int offset;
  int stride;
  Vector2 storage_offset;

  IndexND() {
  }

  IndexND(int x0,
          int x1,
          int y0,
          int y1,
          Vector2 storage_offset = Vector2(0.5f, 0.5f)) {
    x[0] = x0;
    x[1] = x1;
    y[0] = y0;
    y[1] = y1;
    i = x[0];
    j = y[0];
    // offset = 0;
    stride = y[1] - y[0];
    this->storage_offset = storage_offset;
  }

  IndexND(Vector2i start,
          Vector2i end,
          Vector2 storage_offset = Vector2(0.5f, 0.5f)) {
    x[0] = start[0];
    x[1] = end[0];
    y[0] = start[1];
    y[1] = end[1];
    i = x[0];
    j = y[0];
    // offset = 0;
    stride = y[1] - y[0];
    this->storage_offset = storage_offset;
  }

  IndexND(int i, int j) {
    this->i = i;
    this->j = j;
  }

  void next() {
    j++;
    // offset++;
    if (j == y[1]) {
      j = y[0];
      i++;
      if (i == x[1]) {
      }
    }
  }

  Index operator++() {
    this->next();
    return *this;
  }

  bool operator==(const IndexND<2> &o) const {
    return (i == o.i && j == o.j);
  }

  bool operator!=(const IndexND<2> &o) const {
    return !(i == o.i && j == o.j);
  }

  Index &to_end() {
    i = x[1];
    j = y[0];
    // offset = (x[1] - x[0]) * (y[1] - y[0]);
    return *this;
  }

  const Index &operator*() const {
    return *this;
  }

  Index &operator*() {
    return *this;
  }

  int operator[](int c) {
    return *(&i + c);
  }

  int operator[](int c) const {
    return *(&i + c);
  }

  Index neighbour(int di, int dj) const {
    Index i = *this;
    i.i += di;
    i.j += dj;
    return i;
  }

  Index neighbour(Vector2i d) const {
    Index i = *this;
    i.i += d.x;
    i.j += d.y;
    return i;
  }

  Index operator+(Vector2i d) const {
    return neighbour(d);
  }

  Vector2 get_pos() const {
    return Vector2((real)i + storage_offset.x, (real)j + storage_offset.y);
  }

  Vector2i get_ipos() const {
    return Vector2i(i, j);
  }
};

typedef IndexND<2> Index2D;

template <>
class RegionND<2> {
 private:
  int x[2], y[2];
  Index2D index_begin;
  Index2D index_end;
  Vector2 storage_offset;

 public:
  using Region = RegionND<2>;

  RegionND() {
  }

  RegionND(int x0,
           int x1,
           int y0,
           int y1,
           Vector2 storage_offset = Vector2(0.5f, 0.5f)) {
    x[0] = x0;
    x[1] = x1;
    y[0] = y0;
    y[1] = y1;
    index_begin = Index2D(x0, x1, y0, y1, storage_offset);
    index_end = Index2D(x0, x1, y0, y1, storage_offset).to_end();
    this->storage_offset = storage_offset;
  }

  RegionND(Vector2i start,
           Vector2i end,
           Vector2 storage_offset = Vector2(0.5f, 0.5f)) {
    x[0] = start[0];
    x[1] = end[0];
    y[0] = start[1];
    y[1] = end[1];
    index_begin = Index2D(start, end, storage_offset);
    index_end = Index2D(start, end, storage_offset).to_end();
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

typedef RegionND<2> Region2D;

template <typename T>
class ArrayND<2, T> {
 protected:
  Region2D region;
  typedef typename std::vector<T>::iterator iterator;
  int size;
  Vector2i res;
  Vector2 storage_offset = Vector2(0.5f, 0.5f);  // defualt : center storage
 public:
  std::vector<T> data;
  template <typename S>
  using Array2D = ArrayND<2, S>;

  template <typename P>
  friend Array2D<T> operator*(const P &b, const Array2D<T> &a);

  int get_size() const {
    return size;
  }

  const Region2D &get_region() const {
    return region;
  }

  ArrayND(const Vector2i &res,
          T init = T(0),
          Vector2 storage_offset = Vector2(0.5f)) {
    initialize(res, init, storage_offset);
  }

  void initialize(const Vector2i &res,
                  T init = T(0),
                  Vector2 storage_offset = Vector2(0.5f)) {
    this->res = res;
    region = Region2D(0, res[0], 0, res[1], storage_offset);
    size = res[0] * res[1];
    data = std::vector<T>(size, init);
    this->storage_offset = storage_offset;
  }

  Array2D<T> same_shape(T init) const {
    return ArrayND<2, T>(res, init, storage_offset);
  }

  Array2D<T> same_shape() const {
    return ArrayND<2, T>(res);
  }

  ArrayND(const Array2D<T> &arr) : ArrayND(arr.res) {
    this->data = arr.data;
    this->storage_offset = arr.storage_offset;
  }

  template <typename P>
  Array2D<T> operator*(const P &b) const {
    Array2D<T> o(res);
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
    Array2D<T> o(res);
    assert(same_dim(b));
    for (int i = 0; i < size; i++) {
      o.data[i] = data[i] + b.data[i];
    }
    return o;
  }

  Array2D<T> operator-(const Array2D<T> &b) const {
    Array2D<T> o(res);
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
    this->res = arr.res;
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

  ArrayND() {
    res = Vector2i(0);
    size = 0;
    data.resize(0);
  }

  ~ArrayND() {
  }

  void reset(T a) {
    for (int i = 0; i < size; i++) {
      data[i] = a;
    }
  }

  void reset_zero() {
    memset(&data[0], 0, sizeof(T) * data.size());
  }

  bool same_dim(const Array2D<T> &arr) const {
    return res == arr.res;
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
    Array2D<T> o(res);
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
    return &data[0] + i * res[1];
  }

  const T *operator[](int i) const {
    return &data[0] + i * res[1];
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

  T sum() const {
    T ret = 0;
    for (int i = 0; i < size; i++) {
      ret += data[i];
    }
    return ret;
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

  template <typename TT = T,
            typename std::enable_if_t<!std::is_class<TT>::value, int> = 0>
  void print(std::string name = "", const char *temp = "%f") const {
    if (name.size())
      printf("%s[%dx%d]=", name.c_str(), res[0], res[1]);
    printf("\n");
    for (int j = res[1] - 1; j >= 0; j--) {
      for (int i = 0; i < res[0]; i++) {
        printf(temp, (*this)[i][j]);
        printf(" ");
      }
      printf("\n");
    }
    printf("\n");
  }

  template <typename TT = T, typename TS = typename TT::ScalarType>
  void print(std::string name = "", const char *temp = "%f") const {
    if (name.size())
      printf("%s[%dx%d]=", name.c_str(), res[0], res[1]);
    printf("\n");
    for (int j = res[1] - 1; j >= 0; j--) {
      for (int i = 0; i < res[0]; i++) {
        printf("(");
        for (int k = 0; k < TT::D; k++) {
          printf(temp, (*this)[i][j][k]);
          if (k != TT::D - 1) {
            printf(", ");
          }
        }
        printf(") ");
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
      data[i] = sinf(s * i + 231.0_f);
    }
  }

  bool inside(int i, int j) const {
    return 0 <= i && i < res[0] && 0 <= j && j < res[1];
  }

  bool inside(const Vector2i &pos) const {
    return inside(pos[0], pos[1]);
  }

  bool inside(const Index2D &index) const {
    return inside(index.i, index.j);
  }

  T sample(real x, real y) const {
    x = clamp(x - storage_offset.x, 0.0_f, res[0] - 1.0_f - eps);
    y = clamp(y - storage_offset.y, 0.0_f, res[1] - 1.0_f - eps);
    int x_i = clamp(int(x), 0, res[0] - 2);
    int y_i = clamp(int(y), 0, res[1] - 2);
    real x_r = x - x_i;
    real y_r = y - y_i;
    return lerp(x_r, lerp(y_r, get(x_i, y_i), get(x_i, y_i + 1)),
                lerp(y_r, get(x_i + 1, y_i), get(x_i + 1, y_i + 1)));
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
    x = x * res[0];
    y = y * res[1];
    return sample(x, y);
  }

  T sample_relative_coord(const Vector2 &vec) const {
    real x = vec.x * res[0];
    real y = vec.y * res[1];
    return sample(x, y);
  }

  auto begin() const {
    return data.cbegin();
  }

  auto end() const {
    return data.cend();
  }

  auto begin() {
    return data.begin();
  }

  auto end() {
    return data.end();
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

  Vector2i get_res() const {
    return res;
  }

  int get_width() const {
    return res[0];
  }

  int get_height() const {
    return res[1];
  }

  bool empty() const {
    return !(res[0] > 0 && res[1] > 0);
  }

  T get_average() const {
    T sum(0);
    for (int i = 0; i < res[0]; i++) {
      for (int j = 0; j < res[1]; j++) {
        sum += get(i, j);
      }
    }
    return 1.0_f / size * sum;
  }

  bool inside(const Vector2 &pos, real tolerance = 1e-4f) const {
    return (-tolerance <= pos.x && pos.x <= res[0] + tolerance &&
            -tolerance <= pos.y && pos.y < res[1] + tolerance);
  }

  Region2D get_rasterization_region(Vector2 pos, int half_extent) const {
    int x = (int)floor(pos.x - storage_offset.x);
    int y = (int)floor(pos.y - storage_offset.y);
    return Region2D(std::max(0, x - half_extent + 1),
                    std::min(res[0], x + half_extent + 1),
                    std::max(0, y - half_extent + 1),
                    std::min(res[1], y + half_extent + 1), storage_offset);
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
    Array2D<T> out(Vector2i(width, height));
    Vector2 actual_size;
    if (storage_offset == Vector2(0.0_f, 0.0_f)) {
      actual_size = Vector2(this->res[0] - 1, this->res[1] - 1);
    } else {
      actual_size = Vector2(this->res[0], this->res[1]);
    }

    Vector2 scale_factor = actual_size / res.cast<real>();

    for (auto &ind : Region2D(0, res[0], 0, res[1], Vector2(0.5f, 0.5f))) {
      Vector2 p = scale_factor * ind.get_pos();
      out[ind] = sample(p);
    }
    return out;
  }

  Array2D<T> rasterize_scale(int width, int height, int scale) {
    Array2D<T> out(Vector2i(width, height));
    for (auto &ind : out.get_region()) {
      out[ind] = (*this)[ind.i / scale][ind.j / scale];
    }
    return out;
  }

  const std::vector<T> &get_data() const {
    return this->data;
  }

  static constexpr int get_dim() {
    return 2;
  }

  void flip(int axis) {
    if (axis == 0) {
      for (int i = 0; i < res[0] / 2; i++) {
        for (int j = 0; j < res[1]; j++) {
          std::swap((*this)[i][j], (*this)[res[0] - 1 - i][j]);
        }
      }
    } else {
      for (int i = 0; i < res[0]; i++) {
        for (int j = 0; j < res[1] / 2; j++) {
          std::swap((*this)[i][j], (*this)[i][res[1] - 1 - j]);
        }
      }
    }
  }

  // TODO: finally we are going to need a binary serializer

  void write_to_disk(const std::string &fn) {
    FILE *f = fopen(fn.c_str(), "wb");
    fwrite(&res[0], sizeof(res[0]), 1, f);
    fwrite(&res[1], sizeof(res[1]), 1, f);
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
    ret = fread(&res[0], sizeof(res[0]), 1, f);
    if (ret != 1) {
      return false;
    }
    ret = fread(&res[1], sizeof(res[1]), 1, f);
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
    initialize(res, T(0), storage_offset);
    ret = (int)std::fread(&data[0], sizeof(data[0]), size, f);
    if (ret != (std::size_t)size) {
      return false;
    }
    fclose(f);
    return true;
  }

  ArrayND(const std::string &filename) {
    load_image(filename);
  }

  void load_image(const std::string &filename, bool linearize = true);

  void set_pixel(real x, real y, const T &pixel) {
    x *= this->res[0];
    y *= this->res[1];
    x -= 0.5f;
    y -= 0.5f;
    int int_x = (int)x;
    int int_y = (int)y;
    if (int_x < 0 || int_x >= this->res[0] || int_y < 0 ||
        int_y >= this->res[1])
      return;
    this->operator[](int_x)[int_y] = pixel;
  }

  T sample_as_texture(real x, real y, bool interp = true) {
    x *= this->res[0];
    y *= this->res[1];
    x -= 0.5f;
    y -= 0.5f;
    x = clamp(x, 0.0_f, this->res[0] - 1.0_f);
    y = clamp(y, 0.0_f, this->res[1] - 1.0_f);
    int ix = clamp(int(x), 0, this->res[0] - 2);
    int iy = clamp(int(y), 0, this->res[1] - 2);
    if (!interp) {
      x = real(ix);
      y = real(iy);
    }
    T x_0 = lerp(y - iy, (*this)[ix][iy], (*this)[ix][iy + 1]);
    T x_1 = lerp(y - iy, (*this)[ix + 1][iy], (*this)[ix + 1][iy + 1]);
    return lerp(x - ix, x_0, x_1);
  }

  void write_as_image(const std::string &filename);

  void write_text(const std::string &font_fn,
                  const std::string &content,
                  real size,
                  int dx,
                  int dy,
                  T color = T(1.0_f));
};

template <typename T>
using Array2D = ArrayND<2, T>;

template <typename T, typename P>
inline Array2D<T> operator*(const P &b, const Array2D<T> &a) {
  Array2D<T> o(a.res);
  for (int i = 0; i < a.size; i++) {
    o.data[i] = b * a.data[i];
  }
  return o;
}

template <typename T>
inline void print(const Array2D<T> &arr) {
  arr.print("");
}

TI_NAMESPACE_END
