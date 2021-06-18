/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <cmath>
#include <type_traits>
#include <functional>
#include <vector>
#include <array>
#include "taichi/common/core.h"
#include "scalar.h"
#include "array_fwd.h"
TI_NAMESPACE_BEGIN

// Instruction Set Extension

enum class InstSetExt { None };

constexpr InstSetExt default_instruction_set = InstSetExt::None;

/////////////////////////////////////////////////////////////////
/////              N dimensional Vector
/////////////////////////////////////////////////////////////////

template <int dim, typename T, InstSetExt ISE>
struct VectorNDBase {
  static constexpr bool simd = false;
  static constexpr int storage_elements = dim;
  T d[dim];
};

template <typename T, InstSetExt ISE>
struct VectorNDBase<1, T, ISE> {
  static constexpr bool simd = false;
  static constexpr int storage_elements = 1;
  union {
    T d[1];
    struct {
      T x;
    };
  };
};

template <typename T, InstSetExt ISE>
struct VectorNDBase<2, T, ISE> {
  static constexpr bool simd = false;
  static constexpr int storage_elements = 2;
  union {
    T d[2];
    struct {
      T x, y;
    };
  };
};

template <typename T, InstSetExt ISE>
struct VectorNDBase<3, T, ISE> {
  static constexpr bool simd = false;
  static constexpr int storage_elements = 3;
  union {
    T d[3];
    struct {
      T x, y, z;
    };
  };
};

template <typename T, InstSetExt ISE>
struct VectorNDBase<4, T, ISE> {
  static constexpr int storage_elements = 4;
  static constexpr bool simd = false;
  union {
    T d[4];
    struct {
      T x, y, z, w;
    };
  };
};

template <int dim__, typename T, InstSetExt ISE = default_instruction_set>
struct VectorND : public VectorNDBase<dim__, T, ISE> {
  static constexpr int dim = dim__;
  using ScalarType = T;

  using type = T;

  using VectorBase = VectorNDBase<dim, T, ISE>;
  using VectorBase::d;
  static constexpr int storage_elements = VectorBase::storage_elements;

  TI_FORCE_INLINE VectorND() {
    for (int i = 0; i < dim; i++) {
      this->d[i] = T(0);
    }
  }

  static TI_FORCE_INLINE VectorND from_array(const T new_val[dim]) {
    VectorND ret;
    for (int i = 0; i < dim; i++) {
      ret.d[i] = new_val[i];
    }
    return ret;
  }

  template <int dim_, typename T_, InstSetExt ISE_>
  explicit TI_FORCE_INLINE VectorND(const VectorND<dim_, T_, ISE_> &o)
      : VectorND() {
    for (int i = 0; i < std::min(dim_, dim__); i++) {
      d[i] = o[i];
    }
  }

  explicit TI_FORCE_INLINE VectorND(const std::array<T, dim> &o) {
    for (int i = 0; i < dim; i++) {
      d[i] = o[i];
    }
  }

  template <typename T_ = T,
            typename std::enable_if_t<std::is_same<T_, int>::value, int> = 0>
  VectorND(const TIndex<dim> &ind);

  // Vector initialization
  template <typename F,
            std::enable_if_t<std::is_same<F, VectorND>::value, int> = 0>
  explicit TI_FORCE_INLINE VectorND(const F &f) {
    for (int i = 0; i < dim; i++)
      this->d[i] = f[i];
  }

  // Scalar initialization
  template <typename F, std::enable_if_t<std::is_same<F, T>::value, int> = 0>
  explicit TI_FORCE_INLINE VectorND(const F &f) {
    for (int i = 0; i < dim; i++)
      this->d[i] = f;
  }

  // Function intialization
  template <
      typename F,
      std::enable_if_t<std::is_convertible<F, std::function<T(int)>>::value,
                       int> = 0>
  explicit TI_FORCE_INLINE VectorND(const F &f) {
    for (int i = 0; i < dim; i++)
      this->d[i] = f(i);
  }

  template <int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE>
  explicit TI_FORCE_INLINE VectorND(T v) {
    for (int i = 0; i < dim; i++) {
      this->d[i] = v;
    }
  }

  explicit TI_FORCE_INLINE VectorND(T v0, T v1) {
    static_assert(dim == 2, "Vector dim must be 2");
    this->d[0] = v0;
    this->d[1] = v1;
  }

  // All except Vector3f
  template <int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE>
  explicit TI_FORCE_INLINE VectorND(T v0, T v1, T v2) {
    static_assert(dim == 3, "Vector dim must be 3");
    this->d[0] = v0;
    this->d[1] = v1;
    this->d[2] = v2;
  }

  // All except Vector3f, Vector4f
  template <int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE>
  explicit TI_FORCE_INLINE VectorND(T v0, T v1, T v2, T v3) {
    static_assert(dim == 4, "Vector dim must be 4");
    this->d[0] = v0;
    this->d[1] = v1;
    this->d[2] = v2;
    this->d[3] = v3;
  }

  // Vector extension
  template <int dim_ = dim, std::enable_if_t<(dim_ > 1), int> = 0>
  explicit TI_FORCE_INLINE VectorND(const VectorND<dim - 1, T, ISE> &o,
                                    T extra) {
    for (int i = 0; i < dim_ - 1; i++) {
      this->d[i] = o[i];
    }
    this->d[dim - 1] = extra;
  }

  template <typename T_>
  explicit TI_FORCE_INLINE VectorND(const std::vector<T_> &o) {
    if (o.size() != dim) {
      TI_ERROR("Dimension mismatch: " + std::to_string(dim) + " v.s. " +
               std::to_string((int)o.size()));
    }
    for (int i = 0; i < dim; i++)
      this->d[i] = T(o[i]);
  }

  TI_FORCE_INLINE T &operator[](int i) {
    return this->d[i];
  }

  TI_FORCE_INLINE const T &operator[](int i) const {
    return this->d[i];
  }

  TI_FORCE_INLINE T &operator()(int i) {
    return d[i];
  }

  TI_FORCE_INLINE const T &operator()(int i) const {
    return d[i];
  }

  TI_FORCE_INLINE T dot(VectorND<dim, T, ISE> o) const {
    T ret = T(0);
    for (int i = 0; i < dim; i++)
      ret += this->d[i] * o[i];
    return ret;
  }

  template <
      typename F,
      std::enable_if_t<std::is_convertible<F, std::function<T(int)>>::value,
                       int> = 0>
  TI_FORCE_INLINE VectorND &set(const F &f) {
    for (int i = 0; i < dim; i++)
      this->d[i] = f(i);
    return *this;
  }

  TI_FORCE_INLINE auto map(T(f)(T)) const
      -> VectorND<dim, decltype(f(T(0))), ISE> {
    VectorND<dim, decltype(f(T(0))), ISE> ret;
    for (int i = 0; i < dim; i++)
      ret[i] = f(this->d[i]);
    return ret;
  }

  TI_FORCE_INLINE VectorND &operator=(const VectorND &o) {
    memcpy(this, &o, sizeof(*this));
    return *this;
  }

  // Non-SIMD cases
  template <int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE>
  TI_FORCE_INLINE VectorND operator+(const VectorND &o) const {
    return VectorND([=](int i) { return this->d[i] + o[i]; });
  }

  template <int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE>
  TI_FORCE_INLINE VectorND operator-(const VectorND &o) const {
    return VectorND([=](int i) { return this->d[i] - o[i]; });
  }

  template <int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE>
  TI_FORCE_INLINE VectorND operator*(const VectorND &o) const {
    return VectorND([=](int i) { return this->d[i] * o[i]; });
  }

  template <int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE>
  TI_FORCE_INLINE VectorND operator/(const VectorND &o) const {
    return VectorND([=](int i) { return this->d[i] / o[i]; });
  }

  template <int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE>
  TI_FORCE_INLINE VectorND operator%(const VectorND &o) const {
    return VectorND([=](int i) { return this->d[i] % o[i]; });
  }

  // Inplace operations
  TI_FORCE_INLINE VectorND &operator+=(const VectorND &o) {
    (*this) = (*this) + o;
    return *this;
  }

  TI_FORCE_INLINE VectorND &operator-=(const VectorND &o) {
    (*this) = (*this) - o;
    return *this;
  }

  TI_FORCE_INLINE VectorND &operator*=(const VectorND &o) {
    (*this) = (*this) * o;
    return *this;
  }

  TI_FORCE_INLINE VectorND &operator*=(const T &o) {
    (*this) = (*this) * o;
    return *this;
  }

  TI_FORCE_INLINE VectorND &operator/=(const VectorND &o) {
    (*this) = (*this) / o;
    return *this;
  }

  TI_FORCE_INLINE VectorND &operator/=(const T &o) {
    (*this) = (*this) / o;
    return *this;
  }

  TI_FORCE_INLINE VectorND operator-() const {
    return VectorND([=](int i) { return -this->d[i]; });
  }

  TI_FORCE_INLINE bool operator==(const VectorND &o) const {
    for (int i = 0; i < dim; i++)
      if (this->d[i] != o[i])
        return false;
    return true;
  }

  TI_FORCE_INLINE bool operator<(const VectorND &o) const {
    for (int i = 0; i < dim; i++)
      if (this->d[i] >= o[i])
        return false;
    return true;
  }

  TI_FORCE_INLINE bool operator<=(const VectorND &o) const {
    for (int i = 0; i < dim; i++)
      if (this->d[i] > o[i])
        return false;
    return true;
  }

  TI_FORCE_INLINE bool operator>(const VectorND &o) const {
    for (int i = 0; i < dim; i++)
      if (this->d[i] <= o[i])
        return false;
    return true;
  }

  TI_FORCE_INLINE bool operator>=(const VectorND &o) const {
    for (int i = 0; i < dim; i++)
      if (this->d[i] < o[i])
        return false;
    return true;
  }

  TI_FORCE_INLINE bool operator==(const std::vector<T> &o) const {
    if (o.size() != dim)
      return false;
    for (int i = 0; i < dim; i++)
      if (this->d[i] != o[i])
        return false;
    return true;
  }

  TI_FORCE_INLINE bool operator!=(const VectorND &o) const {
    for (int i = 0; i < dim; i++)
      if (this->d[i] != o[i])
        return true;
    return false;
  }

  TI_FORCE_INLINE VectorND abs() const {
    return VectorND([&](int i) { return std::abs(d[i]); });
  }

  TI_FORCE_INLINE VectorND floor() const {
    return VectorND([&](int i) { return std::floor(d[i]); });
  }

  TI_FORCE_INLINE VectorND sin() const {
    return VectorND([&](int i) { return std::sin(d[i]); });
  }

  TI_FORCE_INLINE VectorND cos() const {
    return VectorND([&](int i) { return std::cos(d[i]); });
  }

  TI_FORCE_INLINE VectorND fract() const {
    return VectorND([&](int i) { return taichi::fract(d[i]); });
  }

  TI_FORCE_INLINE VectorND clamp() const {
    return VectorND([&](int i) { return taichi::clamp(d[i]); });
  }

  TI_FORCE_INLINE VectorND clamp(const T &a, T &b) const {
    return VectorND([&](int i) { return taichi::clamp(d[i], a, b); });
  }

  TI_FORCE_INLINE VectorND clamp(const VectorND &a, const VectorND &b) const {
    return VectorND([&](int i) { return taichi::clamp(d[i], a[i], b[i]); });
  }

  TI_FORCE_INLINE T min() const {
    T ret = this->d[0];
    for (int i = 1; i < dim; i++) {
      ret = std::min(ret, this->d[i]);
    }
    return ret;
  }

  TI_FORCE_INLINE T max() const {
    T ret = this->d[0];
    for (int i = 1; i < dim; i++) {
      ret = std::max(ret, this->d[i]);
    }
    return ret;
  }

  TI_FORCE_INLINE T abs_max() const {
    T ret = std::abs(this->d[0]);
    for (int i = 1; i < dim; i++) {
      ret = std::max(ret, std::abs(this->d[i]));
    }
    return ret;
  }

  template <typename G>
  TI_FORCE_INLINE VectorND<dim, G, ISE> cast() const {
    return VectorND<dim, G, ISE>(
        [this](int i) { return static_cast<G>(this->d[i]); });
  }

  void print() const {
    for (int i = 0; i < dim; i++) {
      std::cout << this->d[i] << " ";
    }
    std::cout << std::endl;
  }

  template <int a,
            int b,
            int c,
            int d,
            int dim_ = dim,
            typename T_ = T,
            InstSetExt ISE_ = ISE>
  TI_FORCE_INLINE VectorND permute() const {
    return VectorND(this->d[a], this->d[b], this->d[c], this->d[d]);
  }

  template <int a, int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE>
  TI_FORCE_INLINE VectorND broadcast() const {
    return permute<a, a, a, a>();
  }

  template <int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE>
  TI_FORCE_INLINE T length2() const {
    T ret = 0;
    for (int i = 0; i < dim; i++) {
      ret += this->d[i] * this->d[i];
    }
    return ret;
  }

  TI_FORCE_INLINE auto length() const {
    return std::sqrt(length2());
  }

  bool is_normal() const {
    for (int i = 0; i < dim; i++) {
      if (!taichi::is_normal(this->d[i]))
        return false;
    }
    return true;
  }

  bool abnormal() const {
    return !this->is_normal();
  }

  static VectorND rand() {
    VectorND ret;
    for (int i = 0; i < dim; i++) {
      ret[i] = taichi::rand();
    }
    return ret;
  }

  TI_FORCE_INLINE T sum() const {
    T ret = this->d[0];
    for (int i = 1; i < dim; i++) {
      ret += this->d[i];
    }
    return ret;
  }

  TI_FORCE_INLINE T average() const {
    return (T(1.0) / dim) * sum();
  }

  TI_FORCE_INLINE T prod() const {
    T ret = this->d[0];
    for (int i = 1; i < dim; i++) {
      ret *= this->d[i];
    }
    return ret;
  }

  TI_FORCE_INLINE VectorND pow(T index) const {
    VectorND ret;
    for (int i = 0; i < dim; i++) {
      ret[i] = std::pow(this->d[i], index);
    }
    return ret;
  }

  TI_FORCE_INLINE static VectorND axis(int i) {
    VectorND ret(0);
    ret[i] = 1;
    return ret;
  }

  TI_IO_DECL {
    if (TI_SERIALIZER_IS(TextSerializer)) {
      std::string ret = "(";
      for (int i = 0; i < dim - 1; i++) {
        ret += fmt::format("{}, ", d[i]);
      }
      ret += fmt::format("{}", d[dim - 1]);
      ret += ")";
      serializer("vec", ret);
    } else {
      TI_IO(d);
    }
  }

  TI_FORCE_INLINE operator std::array<T, dim>() const {
    std::array<T, dim> arr;
    for (int i = 0; i < dim; i++) {
      arr[i] = d[i];
    }
    return arr;
  }
};

template <typename T, int dim, InstSetExt ISE = default_instruction_set>
using TVector = VectorND<dim, T, ISE>;

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE VectorND<dim, T, ISE> operator
    *(T a, const VectorND<dim, T, ISE> &v) {
  return VectorND<dim, T, ISE>(a) * v;
}

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE VectorND<dim, T, ISE> operator*(const VectorND<dim, T, ISE> &v,
                                                T a) {
  return a * v;
}

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE VectorND<dim, T, ISE> operator/(
    T a,
    const VectorND<dim, T, ISE> &v) {
  return VectorND<dim, T, ISE>(a) / v;
}

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE VectorND<dim, T, ISE> operator/(const VectorND<dim, T, ISE> &v,
                                                T a) {
  return v / VectorND<dim, T, ISE>(a);
}

template <typename T>
TI_FORCE_INLINE std::array<T, 1> to_std_array(const TVector<T, 1> &v) {
  return std::array<T, 1>{v[0]};
}

template <typename T>
TI_FORCE_INLINE std::array<T, 2> to_std_array(const TVector<T, 2> &v) {
  return std::array<T, 2>{v[0], v[1]};
}

template <typename T>
TI_FORCE_INLINE std::array<T, 3> to_std_array(const TVector<T, 3> &v) {
  return std::array<T, 3>{v[0], v[1], v[2]};
}

template <typename T>
TI_FORCE_INLINE std::array<T, 4> to_std_array(const TVector<T, 4> &v) {
  return std::array<T, 4>{v[0], v[1], v[2], v[3]};
}

using Vector1 = VectorND<1, real, default_instruction_set>;
using Vector2 = VectorND<2, real, default_instruction_set>;
using Vector3 = VectorND<3, real, default_instruction_set>;
using Vector4 = VectorND<4, real, default_instruction_set>;

using Vector1f = VectorND<1, float32, default_instruction_set>;
using Vector2f = VectorND<2, float32, default_instruction_set>;
using Vector3f = VectorND<3, float32, default_instruction_set>;
using Vector4f = VectorND<4, float32, default_instruction_set>;

using Vector1d = VectorND<1, float64, default_instruction_set>;
using Vector2d = VectorND<2, float64, default_instruction_set>;
using Vector3d = VectorND<3, float64, default_instruction_set>;
using Vector4d = VectorND<4, float64, default_instruction_set>;

using Vector1i = VectorND<1, int, default_instruction_set>;
using Vector2i = VectorND<2, int, default_instruction_set>;
using Vector3i = VectorND<3, int, default_instruction_set>;
using Vector4i = VectorND<4, int, default_instruction_set>;

template <typename T>
TI_FORCE_INLINE T fused_mul_add(const T &a, const T &b, const T &c) {
  return a * b + c;
}

/////////////////////////////////////////////////////////////////
/////              N dimensional Matrix
/////////////////////////////////////////////////////////////////

template <int dim__, typename T, InstSetExt ISE = default_instruction_set>
struct MatrixND {
  static constexpr int dim = dim__;

  using ScalarType = T;

  using Vector = VectorND<dim, T, ISE>;
  Vector d[dim];

  static constexpr InstSetExt ise = ISE;
  using type = T;

  TI_FORCE_INLINE MatrixND() {
    for (int i = 0; i < dim; i++) {
      d[i] = VectorND<dim, T, ISE>();
    }
  }

  template <int dim_, typename T_, InstSetExt ISE_>
  TI_FORCE_INLINE explicit MatrixND(const MatrixND<dim_, T_, ISE_> &o)
      : MatrixND() {
    for (int i = 0; i < std::min(dim_, dim__); i++) {
      for (int j = 0; j < std::min(dim_, dim__); j++) {
        d[i][j] = o[i][j];
      }
    }
  }

  TI_FORCE_INLINE MatrixND(T v) : MatrixND() {
    for (int i = 0; i < dim; i++) {
      d[i][i] = v;
    }
  }

  TI_FORCE_INLINE MatrixND(const MatrixND &o) {
    *this = o;
  }

  // Diag
  TI_FORCE_INLINE explicit MatrixND(Vector v) : MatrixND() {
    for (int i = 0; i < dim; i++)
      this->d[i][i] = v[i];
  }

  TI_FORCE_INLINE explicit MatrixND(Vector v0, Vector v1) {
    static_assert(dim == 2, "Matrix dim must be 2");
    this->d[0] = v0;
    this->d[1] = v1;
  }

  TI_FORCE_INLINE explicit MatrixND(Vector v0, Vector v1, Vector v2) {
    static_assert(dim == 3, "Matrix dim must be 3");
    this->d[0] = v0;
    this->d[1] = v1;
    this->d[2] = v2;
  }

  TI_FORCE_INLINE explicit MatrixND(Vector v0,
                                    Vector v1,
                                    Vector v2,
                                    Vector v3) {
    static_assert(dim == 4, "Matrix dim must be 4");
    this->d[0] = v0;
    this->d[1] = v1;
    this->d[2] = v2;
    this->d[3] = v3;
  }

  // Function intialization
  template <
      typename F,
      std::enable_if_t<std::is_convertible<
                           F,
                           std::function<VectorND<dim__, T, ISE>(int)>>::value,
                       int> = 0>
  TI_FORCE_INLINE explicit MatrixND(const F &f) {
    for (int i = 0; i < dim; i++)
      this->d[i] = f(i);
  }

  template <
      typename F,
      std::enable_if_t<std::is_convertible<
                           F,
                           std::function<VectorND<dim__, T, ISE>(int)>>::value,
                       int> = 0>
  TI_FORCE_INLINE MatrixND &set(const F &f) {
    for (int i = 0; i < dim; i++)
      this->d[i] = f(i);
    return *this;
  }

  TI_FORCE_INLINE MatrixND &operator=(const MatrixND &o) {
    for (int i = 0; i < dim; i++) {
      this->d[i] = o[i];
    }
    return *this;
  }

  TI_FORCE_INLINE VectorND<dim, T, ISE> &operator[](int i) {
    return d[i];
  }

  TI_FORCE_INLINE T &operator()(int i, int j) {
    return d[j][i];
  }

  TI_FORCE_INLINE const T &operator()(int i, int j) const {
    return d[j][i];
  }

  TI_FORCE_INLINE const VectorND<dim, T, ISE> &operator[](int i) const {
    return d[i];
  }

  template <int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE>
  TI_FORCE_INLINE VectorND<dim, T, ISE> operator*(
      const VectorND<dim, T, ISE> &o) const {
    VectorND<dim, T, ISE> ret = d[0] * o[0];
    for (int i = 1; i < dim; i++)
      ret += d[i] * o[i];
    return ret;
  }

  template <int dim_ = dim, typename T_ = T, InstSetExt ISE_ = ISE>
  TI_FORCE_INLINE MatrixND operator*(const MatrixND &o) const {
    MatrixND ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        T tmp = 0;
        for (int k = 0; k < dim; k++) {
          tmp += (*this)[k][j] * o[i][k];
        }
        ret[i][j] = tmp;
      }
    }
    return ret;
  }

  TI_FORCE_INLINE static MatrixND outer_product(Vector column, Vector row) {
    return MatrixND([&](int i) { return column * row[i]; });
  }

  TI_FORCE_INLINE MatrixND operator+(const MatrixND &o) const {
    return MatrixND([=](int i) { return this->d[i] + o[i]; });
  }

  TI_FORCE_INLINE MatrixND operator-(const MatrixND &o) const {
    return MatrixND([=](int i) { return this->d[i] - o[i]; });
  }

  TI_FORCE_INLINE MatrixND &operator+=(const MatrixND &o) {
    return this->set([&](int i) { return this->d[i] + o[i]; });
  }

  TI_FORCE_INLINE MatrixND &operator-=(const MatrixND &o) {
    return this->set([&](int i) { return this->d[i] - o[i]; });
  }

  TI_FORCE_INLINE MatrixND operator-() const {
    return MatrixND([=](int i) { return -this->d[i]; });
  }

  TI_FORCE_INLINE bool operator==(const MatrixND &o) const {
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++)
        if (d[i][j] != o[i][j])
          return false;
    return true;
  }

  TI_FORCE_INLINE bool operator!=(const MatrixND &o) const {
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++)
        if (d[i][j] != o[i][j])
          return true;
    return false;
  }

  TI_FORCE_INLINE T frobenius_norm2() const {
    T sum = d[0].length2();
    for (int i = 1; i < dim; i++) {
      sum += d[i].length2();
    }
    return sum;
  }

  TI_FORCE_INLINE auto frobenius_norm() const {
    return std::sqrt(frobenius_norm2());
  }

  TI_FORCE_INLINE MatrixND transposed() const {
    MatrixND ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        ret[i][j] = d[j][i];
      }
    }
    return ret;
  }

  template <typename G>
  TI_FORCE_INLINE MatrixND<dim, G, ISE> cast() const {
    return MatrixND<dim, G, ISE>(
        [=](int i) { return d[i].template cast<G>(); });
  }

  bool is_normal() const {
    for (int i = 0; i < dim; i++) {
      if (!this->d[i].is_normal())
        return false;
    }
    return true;
  }

  bool abnormal() const {
    return !this->is_normal();
  }

  static MatrixND rand() {
    MatrixND ret;
    for (int i = 0; i < dim; i++) {
      ret[i] = Vector::rand();
    }
    return ret;
  }

  TI_FORCE_INLINE Vector diag() const {
    Vector ret;
    for (int i = 0; i < dim; i++) {
      ret[i] = this->d[i][i];
    }
    return ret;
  }

  TI_FORCE_INLINE T sum() const {
    T ret(0);
    for (int i = 0; i < dim; i++) {
      ret += this->d[i].sum();
    }
    return ret;
  }

  TI_FORCE_INLINE T trace() const {
    return this->diag().sum();
  }

  TI_FORCE_INLINE T tr() const {
    return this->trace();
  }

  TI_FORCE_INLINE MatrixND
  elementwise_product(const MatrixND<dim, T> &o) const {
    MatrixND ret;
    for (int i = 0; i < dim; i++) {
      ret[i] = this->d[i] * o[i];
    }
    return ret;
  }

  TI_FORCE_INLINE static MatrixND identidy() {
    return MatrixND(1.0_f);
  }

  TI_IO_DECL {
    if constexpr (TI_SERIALIZER_IS(TextSerializer)) {
      for (int i = 0; i < dim; i++) {
        std::string line = "[";
        for (int j = 0; j < dim; j++) {
          line += fmt::format("{}   ", d[j][i]);
        }
        line += "]";
        serializer.add_line(line);
      }
    } else {
      TI_IO(d);
    }
  }
};

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE MatrixND<dim, T, ISE> operator
    *(const T a, const MatrixND<dim, T, ISE> &M) {
  MatrixND<dim, T, ISE> ret;
  for (int i = 0; i < dim; i++) {
    ret[i] = a * M[i];
  }
  return ret;
}

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE MatrixND<dim, T, ISE> operator*(const MatrixND<dim, T, ISE> &M,
                                                const T a) {
  return a * M;
}

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE MatrixND<dim, T, ISE> transpose(
    const MatrixND<dim, T, ISE> &mat) {
  return mat.transposed();
}

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE MatrixND<dim, T, ISE> transposed(
    const MatrixND<dim, T, ISE> &mat) {
  return transpose(mat);
}

template <typename T, int dim, InstSetExt ISE = default_instruction_set>
using TMatrix = MatrixND<dim, T, ISE>;

using Matrix2 = MatrixND<2, real, default_instruction_set>;
using Matrix3 = MatrixND<3, real, default_instruction_set>;
using Matrix4 = MatrixND<4, real, default_instruction_set>;

using Matrix2f = MatrixND<2, float32, default_instruction_set>;
using Matrix3f = MatrixND<3, float32, default_instruction_set>;
using Matrix4f = MatrixND<4, float32, default_instruction_set>;

using Matrix2d = MatrixND<2, float64, default_instruction_set>;
using Matrix3d = MatrixND<3, float64, default_instruction_set>;
using Matrix4d = MatrixND<4, float64, default_instruction_set>;

template <typename T, InstSetExt ISE>
TI_FORCE_INLINE real determinant(const MatrixND<2, T, ISE> &mat) {
  return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
}

template <typename T, InstSetExt ISE>
TI_FORCE_INLINE T determinant(const MatrixND<3, T, ISE> &mat) {
  return mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2]) -
         mat[1][0] * (mat[0][1] * mat[2][2] - mat[2][1] * mat[0][2]) +
         mat[2][0] * (mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2]);
}

template <typename T, InstSetExt ISE>
TI_FORCE_INLINE T cross(const VectorND<2, T, ISE> &a,
                        const VectorND<2, T, ISE> &b) {
  return a.x * b.y - a.y * b.x;
}

template <typename T, InstSetExt ISE>
TI_FORCE_INLINE VectorND<3, T, ISE> cross(const VectorND<3, T, ISE> &a,
                                          const VectorND<3, T, ISE> &b) {
  return VectorND<3, T, ISE>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                             a.x * b.y - a.y * b.x);
}

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE T dot(const VectorND<dim, T, ISE> &a,
                      const VectorND<dim, T, ISE> &b) {
  return a.dot(b);
}

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE VectorND<dim, T, ISE> normalize(
    const VectorND<dim, T, ISE> &a) {
  return (T(1) / a.length()) * a;
}

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE VectorND<dim, T, ISE> normalized(
    const VectorND<dim, T, ISE> &a) {
  return normalize(a);
}

TI_FORCE_INLINE float32 length(const float32 &a) {
  return a;
}

TI_FORCE_INLINE float64 length(const float64 &a) {
  return a;
}

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE T length(const VectorND<dim, T, ISE> &a) {
  return a.length();
}

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE T length2(const VectorND<dim, T, ISE> &a) {
  return dot(a, a);
}

TI_FORCE_INLINE float32 length2(const float32 &a) {
  return a * a;
}

TI_FORCE_INLINE float64 length2(const float64 &a) {
  return a * a;
}

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE VectorND<dim, T, ISE> fract(const VectorND<dim, T, ISE> &a) {
  return a.fract();
}

TI_FORCE_INLINE float32 inversed(const float32 &a) {
  return 1.0_f32 / a;
}

TI_FORCE_INLINE float64 inversed(const float64 &a) {
  return 1.0_f64 / a;
}

template <InstSetExt ISE, typename T>
TI_FORCE_INLINE MatrixND<2, T, ISE> inversed(const MatrixND<2, T, ISE> &mat) {
  T det = determinant(mat);
  return static_cast<T>(1) / det *
         MatrixND<2, T, ISE>(VectorND<2, T, ISE>(mat[1][1], -mat[0][1]),
                             VectorND<2, T, ISE>(-mat[1][0], mat[0][0]));
}

template <InstSetExt ISE, typename T>
MatrixND<3, T, ISE> inversed(const MatrixND<3, T, ISE> &mat) {
  T det = determinant(mat);
  return T(1.0) / det *
         MatrixND<3, T, ISE>(
             VectorND<3, T, ISE>(mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2],
                                 mat[2][1] * mat[0][2] - mat[0][1] * mat[2][2],
                                 mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2]),
             VectorND<3, T, ISE>(mat[2][0] * mat[1][2] - mat[1][0] * mat[2][2],
                                 mat[0][0] * mat[2][2] - mat[2][0] * mat[0][2],
                                 mat[1][0] * mat[0][2] - mat[0][0] * mat[1][2]),
             VectorND<3, T, ISE>(
                 mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1],
                 mat[2][0] * mat[0][1] - mat[0][0] * mat[2][1],
                 mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]));
}

template <typename T, InstSetExt ISE>
T determinant(const MatrixND<4, T, ISE> &m) {
  // This function is adopted from GLM
  /*
  ================================================================================
  OpenGL Mathematics (GLM)
  --------------------------------------------------------------------------------
  GLM is licensed under The Happy Bunny License and MIT License

  ================================================================================
  The Happy Bunny License (Modified MIT License)
  --------------------------------------------------------------------------------
  Copyright (c) 2005 - 2014 G-Truc Creation

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  Restrictions:
   By making use of the Software for military purposes, you choose to make a
   Bunny unhappy.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.

  ================================================================================
  The MIT License
  --------------------------------------------------------------------------------
  Copyright (c) 2005 - 2014 G-Truc Creation

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
   */

  T Coef00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
  T Coef02 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
  T Coef03 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

  T Coef04 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
  T Coef06 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
  T Coef07 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

  T Coef08 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
  T Coef10 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
  T Coef11 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

  T Coef12 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
  T Coef14 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
  T Coef15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

  T Coef16 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
  T Coef18 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
  T Coef19 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

  T Coef20 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
  T Coef22 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
  T Coef23 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

  using Vector = VectorND<4, T, ISE>;

  Vector Fac0(Coef00, Coef00, Coef02, Coef03);
  Vector Fac1(Coef04, Coef04, Coef06, Coef07);
  Vector Fac2(Coef08, Coef08, Coef10, Coef11);
  Vector Fac3(Coef12, Coef12, Coef14, Coef15);
  Vector Fac4(Coef16, Coef16, Coef18, Coef19);
  Vector Fac5(Coef20, Coef20, Coef22, Coef23);

  Vector Vec0(m[1][0], m[0][0], m[0][0], m[0][0]);
  Vector Vec1(m[1][1], m[0][1], m[0][1], m[0][1]);
  Vector Vec2(m[1][2], m[0][2], m[0][2], m[0][2]);
  Vector Vec3(m[1][3], m[0][3], m[0][3], m[0][3]);

  Vector Inv0(Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2);
  Vector Inv1(Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4);
  Vector Inv2(Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5);
  Vector Inv3(Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5);

  Vector SignA(+1, -1, +1, -1);
  Vector SignB(-1, +1, -1, +1);
  MatrixND<4, T, ISE> Inverse(Inv0 * SignA, Inv1 * SignB, Inv2 * SignA,
                              Inv3 * SignB);

  Vector Row0(Inverse[0][0], Inverse[1][0], Inverse[2][0], Inverse[3][0]);

  Vector Dot0(m[0] * Row0);
  T Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w);

  return Dot1;
}

template <typename T, InstSetExt ISE>
MatrixND<4, T, ISE> inversed(const MatrixND<4, T, ISE> &m) {
  // This function is copied from GLM
  /*
  ================================================================================
  OpenGL Mathematics (GLM)
  --------------------------------------------------------------------------------
  GLM is licensed under The Happy Bunny License and MIT License

  ================================================================================
  The Happy Bunny License (Modified MIT License)
  --------------------------------------------------------------------------------
  Copyright (c) 2005 - 2014 G-Truc Creation

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  Restrictions:
   By making use of the Software for military purposes, you choose to make a
   Bunny unhappy.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.

  ================================================================================
  The MIT License
  --------------------------------------------------------------------------------
  Copyright (c) 2005 - 2014 G-Truc Creation

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
   */

  T Coef00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
  T Coef02 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
  T Coef03 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

  T Coef04 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
  T Coef06 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
  T Coef07 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

  T Coef08 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
  T Coef10 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
  T Coef11 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

  T Coef12 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
  T Coef14 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
  T Coef15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

  T Coef16 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
  T Coef18 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
  T Coef19 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

  T Coef20 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
  T Coef22 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
  T Coef23 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

  using Vector = VectorND<4, T, ISE>;

  Vector Fac0(Coef00, Coef00, Coef02, Coef03);
  Vector Fac1(Coef04, Coef04, Coef06, Coef07);
  Vector Fac2(Coef08, Coef08, Coef10, Coef11);
  Vector Fac3(Coef12, Coef12, Coef14, Coef15);
  Vector Fac4(Coef16, Coef16, Coef18, Coef19);
  Vector Fac5(Coef20, Coef20, Coef22, Coef23);

  Vector Vec0(m[1][0], m[0][0], m[0][0], m[0][0]);
  Vector Vec1(m[1][1], m[0][1], m[0][1], m[0][1]);
  Vector Vec2(m[1][2], m[0][2], m[0][2], m[0][2]);
  Vector Vec3(m[1][3], m[0][3], m[0][3], m[0][3]);

  Vector Inv0(Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2);
  Vector Inv1(Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4);
  Vector Inv2(Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5);
  Vector Inv3(Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5);

  Vector SignA(+1, -1, +1, -1);
  Vector SignB(-1, +1, -1, +1);
  MatrixND<4, T, ISE> Inverse(Inv0 * SignA, Inv1 * SignB, Inv2 * SignA,
                              Inv3 * SignB);

  Vector Row0(Inverse[0][0], Inverse[1][0], Inverse[2][0], Inverse[3][0]);

  Vector Dot0(m[0] * Row0);
  T Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w);

  T OneOverDeterminant = static_cast<T>(1) / Dot1;

  return Inverse * OneOverDeterminant;
}

template <int dim, typename T, InstSetExt ISE>
TI_FORCE_INLINE MatrixND<dim, T, ISE> inverse(const MatrixND<dim, T, ISE> &m) {
  return inversed(m);
}

TI_FORCE_INLINE Vector3 multiply_matrix4(const Matrix4 &m,
                                         const Vector3 &v,
                                         real w) {
  return Vector3(m * Vector4(v, w));
}

template <int dim>
TI_FORCE_INLINE VectorND<dim, real> transform(const MatrixND<dim + 1, real> &m,
                                              const VectorND<dim, real> &v,
                                              real w = 1.0_f) {
  return VectorND<dim, real>(m * VectorND<dim + 1, real>(v, w));
}

// Type traits

template <typename T>
struct is_vector {
  static constexpr bool value = false;
};

template <int dim, typename T, InstSetExt ISE>
struct is_vector<VectorND<dim, T, ISE>> {
  static constexpr bool value = true;
};

template <typename T>
struct is_matrix {
  static constexpr bool value = false;
};

template <int dim, typename T, InstSetExt ISE>
struct is_matrix<MatrixND<dim, T, ISE>> {
  static constexpr bool value = true;
};

template <int dim, typename T>
TI_FORCE_INLINE VectorND<dim, T> min(const VectorND<dim, T> &a,
                                     const VectorND<dim, T> &b) {
  VectorND<dim, T> ret;
  for (int i = 0; i < dim; i++) {
    ret[i] = std::min(a[i], b[i]);
  }
  return ret;
}

template <int dim, typename T>
TI_FORCE_INLINE VectorND<dim, T> max(const VectorND<dim, T> &a,
                                     const VectorND<dim, T> &b) {
  VectorND<dim, T> ret;
  for (int i = 0; i < dim; i++) {
    ret[i] = std::max(a[i], b[i]);
  }
  return ret;
}

inline Matrix4 matrix4_translate(Matrix4 *transform, const Vector3 &offset) {
  return Matrix4(Vector4(1, 0, 0, 0), Vector4(0, 1, 0, 0), Vector4(0, 0, 1, 0),
                 Vector4(offset, 1.0_f)) *
         *transform;
}

inline Matrix4 matrix_translate(Matrix4 *transform, const Vector3 &offset) {
  return matrix4_translate(transform, offset);
}

inline Matrix3 matrix_translate(Matrix3 *transform, const Vector2 &offset) {
  return Matrix3(Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(offset, 1.0_f)) *
         *transform;
}

inline Matrix3 matrix_scale(Matrix3 *transform, const Vector2 &scales) {
  return Matrix3(Vector3(scales, 1.0_f)) * *transform;
}

inline Matrix4 matrix4_scale(Matrix4 *transform, const Vector3 &scales) {
  return Matrix4(Vector4(scales, 1.0_f)) * *transform;
}

inline Matrix4 matrix_scale(Matrix4 *transform, const Vector3 &scales) {
  return Matrix4(Vector4(scales, 1.0_f)) * *transform;
}

inline Matrix4 matrix4_scale_s(Matrix4 *transform, real s) {
  return matrix4_scale(transform, Vector3(s));
}

// Reference: https://en.wikipedia.org/wiki/Rotation_matrix
inline Matrix4 get_rotation_matrix(Vector3 u, real angle) {
  u = normalized(u);
  real c = cos(angle), s = sin(angle);
  real d = 1 - c;

  auto col0 = Vector4(c + u.x * u.x * d, u.x * u.y * d - u.z * s,
                      u.x * u.z * d + u.y * s, 0.0_f);
  auto col1 = Vector4(u.x * u.y * d + u.z * s, c + u.y * u.y * d,
                      u.y * u.z * d - u.x * s, 0.0_f);
  auto col2 = Vector4(u.x * u.z * d - u.y * s, u.y * u.z * d + u.x * s,
                      c + u.z * u.z * d, 0.0_f);
  auto col3 = Vector4(0.0_f, 0.0_f, 0.0_f, 1.0_f);

  return Matrix4(col0, col1, col2, col3).transposed();
}

inline Matrix4 matrix4_rotate_angle_axis(Matrix4 *transform,
                                         real angle,
                                         const Vector3 &axis) {
  return get_rotation_matrix(axis, angle * (pi / 180.0_f)) * *transform;
}

inline Matrix4 matrix4_rotate_euler(Matrix4 *transform,
                                    const Vector3 &euler_angles) {
  Matrix4 ret = *transform;
  ret = matrix4_rotate_angle_axis(&ret, euler_angles.x,
                                  Vector3(1.0_f, 0.0_f, 0.0_f));
  ret = matrix4_rotate_angle_axis(&ret, euler_angles.y,
                                  Vector3(0.0_f, 1.0_f, 0.0_f));
  ret = matrix4_rotate_angle_axis(&ret, euler_angles.z,
                                  Vector3(0.0_f, 0.0_f, 1.0_f));
  return ret;
}

template <typename T>
inline MatrixND<3, T> cross_product_matrix(const VectorND<3, T> &a) {
  return MatrixND<3, T>(VectorND<3, T>(0, a[2], -a[1]),
                        VectorND<3, T>(-a[2], 0, a[0]),
                        VectorND<3, T>(a[1], -a[0], 0));
};

static_assert(Serializer::has_io<Matrix4>::value, "");
static_assert(Serializer::has_io<const Matrix4>::value, "");
static_assert(Serializer::has_io<const Matrix4 &>::value, "");
static_assert(Serializer::has_io<Matrix4 &>::value, "");
static_assert(
    TextSerializer::has_io<
        const taichi::MatrixND<4, double, (taichi::InstSetExt)3>>::value,
    "");

namespace type {
template <typename T, typename = void>
struct element_;

template <typename T>
struct element_<T, typename std::enable_if_t<std::is_arithmetic<T>::value>> {
  using type = T;
};

template <typename T>
struct element_<T, typename std::enable_if_t<!std::is_arithmetic<T>::value>> {
  using type = typename T::ScalarType;
};

template <typename T>
using element = typename element_<std::decay_t<T>>::type;

template <typename>
struct is_VectorND : public std::false_type {};

template <int N, typename T, InstSetExt ISE>
struct is_VectorND<VectorND<N, T, ISE>> : public std::true_type {};

template <typename>
struct is_MatrixND : public std::false_type {};

template <int N, typename T, InstSetExt ISE>
struct is_MatrixND<MatrixND<N, T, ISE>> : public std::true_type {};
}  // namespace type

TI_NAMESPACE_END
