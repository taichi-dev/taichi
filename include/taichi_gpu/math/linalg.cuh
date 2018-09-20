#pragma once

#define TC_FORCE_INLINE __forceinline__
using real = float;

template <typename T, int dim_>
class TVector {
 public:
  static constexpr int dim = dim_;

  T d[dim];

  TC_FORCE_INLINE __host__ __device__ TVector(T *val) {
    for (int i = 0; i < dim; i++) {
      d[i] = val[i];
    }
  }

  TC_FORCE_INLINE __device__ __host__ T *data() {
    return &d[0];
  }

  template <int dim__ = dim_>
  TC_FORCE_INLINE __host__ __device__ TVector(T x, T y) {
    static_assert(dim__ == 2, "");
    d[0] = x;
    d[1] = y;
  }

  template <int dim__ = dim_>
  TC_FORCE_INLINE __host__ __device__ TVector(T x, T y, T z) {
    static_assert(dim__ == 3, "");
    d[0] = x;
    d[1] = y;
    d[2] = z;
  }

  TC_FORCE_INLINE __host__ __device__ TVector(T x = 0) {
    for (int i = 0; i < dim; i++) {
      d[i] = x;
    }
  }

  TC_FORCE_INLINE __host__ __device__ T operator[](int i) const {
    return d[i];
  }

  TC_FORCE_INLINE __host__ __device__ T &operator[](int i) {
    return d[i];
  }

  TC_FORCE_INLINE __host__ __device__ TVector &operator+=(const TVector &o) {
    for (int i = 0; i < dim; i++) {
      d[i] += o[i];
    }
    return *this;
  }

  TC_FORCE_INLINE __host__ __device__ TVector operator+(const TVector &o) {
    TVector ret;
    for (int i = 0; i < dim; i++) {
      ret[i] = d[i] + o[i];
    }
    return ret;
  }

  TC_FORCE_INLINE __host__ __device__ TVector operator-(const TVector &o) {
    TVector ret;
    for (int i = 0; i < dim; i++) {
      ret[i] = d[i] - o[i];
    }
    return ret;
  }

  TC_FORCE_INLINE __host__ __device__ T length2() const {
    T ret = 0;
    for (int i = 0; i < dim; i++) {
      ret += d[i] * d[i];
    }
    return ret;
  }

  TC_FORCE_INLINE __host__ __device__ T dot(TVector &other) const {
    T ret = 0;
    for (int i = 0; i < dim; i++) {
      ret += d[i] * other[i];
    }
    return ret;
  }

  TC_FORCE_INLINE T __host__ __device__ __host__ prod() {
    T ret = d[0];
    for (int i = 1; i < dim; i++) {
      ret *= d[i];
    }
    return ret;
  }

  TC_FORCE_INLINE T __device__ __host__ sum() {
    T ret = d[0];
    for (int i = 1; i < dim; i++) {
      ret += d[i];
    }
    return ret;
  }
};

template <typename T, int dim>
TC_FORCE_INLINE __device__ TVector<T, dim> operator*(real alpha,
                                                     const TVector<T, dim> &o) {
  TVector<T, dim> ret;
  for (int i = 0; i < dim; i++) {
    ret[i] = alpha * o[i];
  }
  return ret;
}

using Vector2 = TVector<real, 2>;
using Vector3 = TVector<real, 3>;
using Vector4 = TVector<real, 4>;

using Vector2i = TVector<int, 2>;
using Vector3i = TVector<int, 3>;
using Vector4i = TVector<int, 4>;

template <typename T, int dim_>
class TMatrix {
 public:
  static constexpr int dim = dim_;
  using Vector = TVector<T, dim>;

  T d[dim][dim];

  TC_FORCE_INLINE __device__ __host__ T *data() {
    return &d[0][0];
  }

  template <int dim__ = dim_>
  TC_FORCE_INLINE __device__ TMatrix(T a00, T a01, T a10, T a11) {
    static_assert(dim__ == 2, "");
    d[0][0] = a00;
    d[0][1] = a01;
    d[1][0] = a10;
    d[1][1] = a11;
  }

  template <int dim__ = dim_>
  TC_FORCE_INLINE __device__
  TMatrix(T a00, T a01, T a02, T a10, T a11, T a12, T a20, T a21, T a22) {
    static_assert(dim__ == 3, "");
    d[0][0] = a00;
    d[0][1] = a01;
    d[0][2] = a02;
    d[1][0] = a10;
    d[1][1] = a11;
    d[1][2] = a12;
    d[2][0] = a20;
    d[2][1] = a21;
    d[2][2] = a22;
  }

  TC_FORCE_INLINE __host__ __device__ TMatrix(T x = 0) {
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        d[i][j] = (i == j) ? x : 0;
      }
    }
  }

  TC_FORCE_INLINE __host__ __device__ T *operator[](int i) {
    return d[i];
  }

  TC_FORCE_INLINE __host__ __device__ T const *operator[](int i) const {
    return d[i];
  }

  TC_FORCE_INLINE __device__ TMatrix operator*(const TMatrix &o) {
    TMatrix ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        for (int k = 0; k < dim; k++) {
          ret[i][j] += d[i][k] * o[k][j];
        }
      }
    }
    return ret;
  }

  TC_FORCE_INLINE __device__ TMatrix operator+(const TMatrix &o) {
    TMatrix ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        ret[i][j] = d[i][j] + o[i][j];
      }
    }
    return ret;
  }

  TC_FORCE_INLINE __device__ TMatrix operator-(const TMatrix &o) {
    TMatrix ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        ret[i][j] = d[i][j] - o[i][j];
      }
    }
    return ret;
  }

  TC_FORCE_INLINE __device__ Vector operator*(const Vector &v) {
    Vector ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        ret[i] += d[i][j] * v[j];
      }
    }
    return ret;
  }

  TC_FORCE_INLINE __device__ T &operator()(int i, int j) {
    return d[i][j];
  }

  TC_FORCE_INLINE __device__ const T &operator()(int i, int j) const {
    return d[i][j];
  }

  static __device__ TMatrix outer_product(const Vector &col,
                                          const Vector &row) {
    TMatrix ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        ret[i][j] = col[i] * row[j];
      }
    }
    return ret;
  }

  TC_FORCE_INLINE __device__ TMatrix elementwise_dot(TMatrix o) {
    T ret = 0;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        ret += (*this)[i][j] * o[i][j];
      }
    }
    return ret;
  }
};

using Matrix2 = TMatrix<real, 2>;
using Matrix3 = TMatrix<real, 3>;
using Matrix4 = TMatrix<real, 4>;

template <typename T>
TC_FORCE_INLINE __device__ TMatrix<T, 2> transposed(const TMatrix<T, 2> &A) {
  return TMatrix<T, 2>(A[0][0], A[1][0], A[0][1], A[1][1]);
}

template <typename T>
TC_FORCE_INLINE __device__ TMatrix<T, 3> transposed(const TMatrix<T, 3> &A) {
  return TMatrix<T, 3>(A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1],
                       A[0][2], A[1][2], A[2][2]);
}

template <typename T>
TC_FORCE_INLINE __device__ T determinant(const TMatrix<T, 2> &mat) {
  return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
}

template <typename T>
TC_FORCE_INLINE __device__ T determinant(const TMatrix<T, 3> &mat) {
  return mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2]) -
         mat[1][0] * (mat[0][1] * mat[2][2] - mat[2][1] * mat[0][2]) +
         mat[2][0] * (mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2]);
}

template <typename T, int dim>
TC_FORCE_INLINE __device__ TMatrix<T, dim> operator*(T alpha,
                                                     const TMatrix<T, dim> &o) {
  TMatrix<T, dim> ret;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      ret[i][j] = alpha * o[i][j];
    }
  }
  return ret;
}

TC_FORCE_INLINE __device__ real sqr(real x) {
  return x * x;
}

TC_FORCE_INLINE void __device__ polar_decomp(TMatrix<real, 2> &m,
                                             TMatrix<real, 2> &R,
                                             TMatrix<real, 2> &S) {
  auto x = m(0, 0) + m(1, 1);
  auto y = m(1, 0) - m(0, 1);
  auto scale = 1.0f / sqrtf(x * x + y * y);
  auto c = x * scale;
  auto s = y * scale;
  R = TMatrix<real, 2>(c, -s, s, c);
  S = transposed(R) * m;
}

TC_FORCE_INLINE __device__ TMatrix<real, 2> inversed(
    const TMatrix<real, 2> &mat) {
  real det = determinant(mat);
  return (1 / det) *
         TMatrix<real, 2>(mat[1][1], -mat[0][1], -mat[1][0], mat[0][0]);
}

TC_FORCE_INLINE __device__ TMatrix<real, 3> inversed(
    const TMatrix<real, 3> &mat) {
  real det = determinant(mat);
  return 1.0f / det *
         TMatrix<real, 3>(mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2],
                          mat[2][1] * mat[0][2] - mat[0][1] * mat[2][2],
                          mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2],
                          mat[2][0] * mat[1][2] - mat[1][0] * mat[2][2],
                          mat[0][0] * mat[2][2] - mat[2][0] * mat[0][2],
                          mat[1][0] * mat[0][2] - mat[0][0] * mat[1][2],
                          mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1],
                          mat[2][0] * mat[0][1] - mat[0][0] * mat[2][1],
                          mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]);
}

TC_FORCE_INLINE __device__ real sgn(real x) {
  return x > 0 ? 1 : -1;
}
