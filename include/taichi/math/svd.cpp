/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include <implicit_qr_svd/Tools.h>
#include <implicit_qr_svd/ImplicitQRSVD.h>
#include "eigen.h"
#pragma GCC diagnostic pop
#include <taichi/testing.h>
#include "sifakis_svd.h"
#include "svd.h"

//#define TC_USE_EIGEN_SVD

TC_NAMESPACE_BEGIN

template <int dim, typename T>
void eigen_svd(const MatrixND<dim, T> &m,
               MatrixND<dim, T> &u,
               MatrixND<dim, T> &s,
               MatrixND<dim, T> &v) {
  Eigen::Matrix<T, dim, dim> e_m(dim, dim);
  e_m = to_eigen(m);
  Eigen::JacobiSVD<Eigen::Matrix<T, dim, dim>> e_svd(
      e_m, Eigen::ComputeThinU | Eigen::ComputeThinV);
  s = MatrixND<dim, T>(0.0_f);
  for (int i = 0; i < dim; ++i)
    s[i][i] = e_svd.singularValues()(i);
  u = from_eigen<dim, T>(e_svd.matrixU());
  v = from_eigen<dim, T>(e_svd.matrixV());
}

template <int dim, typename T>
TC_FORCE_INLINE void ensure_non_negative_singular_values(MatrixND<dim, T> &u,
                                                         MatrixND<dim, T> &s) {
  for (int i = 0; i < dim; i++) {
    if (s[i][i] < 0) {
      s[i][i] = -s[i][i];
      u[i] *= -1.0_f;
    }
  }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
template <int dim, typename T>
void imp_svd(const MatrixND<dim, T> &m_,
             MatrixND<dim, T> &u,
             MatrixND<dim, T> &s,
             MatrixND<dim, T> &v) {
  using Matrix = MatrixND<dim, T>;
  Matrix m = m_;
  TC_STATIC_IF(dim == 2) {
    if ((m - Matrix(m.diag())).frobenius_norm2() < 1e-7f) {
      s = m;
      u = v = Matrix(1);
      if (abs(s[0][0]) < abs(s[1][1])) {
        std::swap(s[0][0], s[1][1]);
      }
      if (s[0][0] < 0) {
        s = -s;
        u = -u;
      }
    } else {
      JIXIE::singularValueDecomposition(
          *(Eigen::Matrix<T, dim, dim> *)&m, *(Eigen::Matrix<T, dim, dim> *)&u,
          *(Eigen::Matrix<T, dim, 1> *)&s, *(Eigen::Matrix<T, dim, dim> *)&v);
      T s_tmp[]{s[0][1]};
      memset(&s[0][0] + 1, 0, sizeof(T) * 3);
      s[1][1] = s_tmp[0];
    }
  }
  TC_STATIC_ELSE{
      // clang-format off
    TC_STATIC_IF((std::is_same<T, float32>() == true)) {
      SifakisSVD::svd<5>(
           m_(0, 0),
           m_(0, 1),
           m_(0, 2),
           m_(1, 0),
           m_(1, 1),
           m_(1, 2),
           m_(2, 0),
           m_(2, 1),
           m_(2, 2),
           u(0, 0),
           u(0, 1),
           u(0, 2),
           u(1, 0),
           u(1, 1),
           u(1, 2),
           u(2, 0),
           u(2, 1),
           u(2, 2),
           v(0, 0),
           v(0, 1),
           v(0, 2),
           v(1, 0),
           v(1, 1),
           v(1, 2),
           v(2, 0),
           v(2, 1),
           v(2, 2),
           s(0, 0),
           s(1, 1),
           s(2, 2)
      );
      s(0, 1) = s(1, 0) = s(0, 2) = s(2, 0) = s(1, 2) = s(2, 1) = 0;
    } TC_STATIC_ELSE {
      eigen_svd(m_, u, s, v);
    } TC_STATIC_END_IF
// clang-format on
}
TC_STATIC_END_IF
}
#pragma GCC diagnostic pop

template <int dim, typename T>
void svd(const MatrixND<dim, T> &m,
         MatrixND<dim, T> &u,
         MatrixND<dim, T> &sig,
         MatrixND<dim, T> &v) {
  using Matrix = MatrixND<dim, T>;
  using Vector = VectorND<dim, T>;
#ifdef TC_USE_EIGEN_SVD
  eigen_svd(m, u, sig, v);
#else
  TC_STATIC_IF(dim == 3) {
    if ((m - Matrix(Vector(m[0][0], m[1][1], m[2][2]))).frobenius_norm2() <
        static_cast<T>(1e-7)) {
      // QR_SVD crashes in this case...
      sig = m;
      u = v = Matrix(1);
    } else {
      imp_svd(m, u, sig, v);
    }
  }
  TC_STATIC_ELSE {
    imp_svd(m, u, sig, v);
  }
  TC_STATIC_END_IF
#endif
  ensure_non_negative_singular_values(u, sig);
}

template <int dim, typename T>
void svd_rot(const MatrixND<dim, T> &m,
             MatrixND<dim, T> &u,
             MatrixND<dim, T> &sig,
             MatrixND<dim, T> &v) {
  TC_STATIC_IF(dim == 3) {
    /*
    if ((m - Matrix(m.diag())).frobenius_norm2() < static_cast<T>(1e-7)) {
      // QR_SVD crashes in this case...
      sig = m;
      u = v = Matrix(1);
    } else {
     */
    imp_svd(m, u, sig, v);
    //}
  }
  TC_STATIC_ELSE {
    imp_svd(m, u, sig, v);
  }
  TC_STATIC_END_IF
}

template <int dim, typename T>
void qr_decomp(const MatrixND<dim, T> &A,
               MatrixND<dim, T> &q,
               MatrixND<dim, T> &r) {
  TC_STATIC_IF(dim == 2) {
    T a = A[0][0], b = A[0][1], inv_r = 1.0 / std::hypot(a, b);
    a *= inv_r;
    b *= inv_r;
    MatrixND<dim, T> Q(VectorND<dim, T>(a, -b), VectorND<dim, T>(b, a));
    q = Q.transposed();
    r = Q * A;
    for (int i = 0; i < dim; i++) {
      if (r[i][i] < 0) {
        for (int j = 0; j < dim; j++) {
          r[j][i] *= -1;
        }
        q[i] *= -1;
      }
    }
  }
  TC_STATIC_ELSE{TC_NOT_IMPLEMENTED} TC_STATIC_END_IF
}

template <int dim, typename T>
void polar_decomp(const MatrixND<dim, T> &A,
                  MatrixND<dim, T> &r,
                  MatrixND<dim, T> &s) {
  MatrixND<dim, T> u, sig, v;
  svd(A, u, sig, v);
  r = u * transposed(v);
  s = v * sig * transposed(v);
}

void svd_eigen3(void const *A_, void *u_, void *sig_, void *v_) {
  Eigen::Matrix3d A = *reinterpret_cast<Eigen::Matrix3d const *>(A_);
  Eigen::Matrix3d u;
  Eigen::Matrix<double, 3, 1> sig;
  Eigen::Matrix3d v;

  Eigen::Vector3d diagonal = A.diagonal();
  Eigen::Matrix3d Adiag = diagonal.asDiagonal();
  if ((A - Adiag).norm() < 1e-7_f) {
    // QR_SVD crashes in this case...
    sig = diagonal;
    u = v = Eigen::Matrix3d::Identity();
  } else {
    JIXIE::singularValueDecomposition(A, u, sig, v);
  }

  *reinterpret_cast<Eigen::Matrix3d *>(u_) = u;
  *reinterpret_cast<Eigen::Matrix<double, 3, 1> *>(sig_) = sig;
  *reinterpret_cast<Eigen::Matrix3d *>(v_) = v;
}

void svd_eigen2(void const *A_, void *u_, void *sig_, void *v_) {
  Eigen::Matrix2d A = *reinterpret_cast<Eigen::Matrix2d const *>(A_);
  Eigen::Matrix2d u;
  Eigen::Matrix<double, 2, 1> sig;
  Eigen::Matrix2d v;

  Eigen::Vector2d diagonal = A.diagonal();
  Eigen::Matrix2d Adiag = diagonal.asDiagonal();
  if ((A - Adiag).norm() < 1e-7_f) {
    // QR_SVD crashes in this case...
    sig = diagonal;
    u = v = Eigen::Matrix2d::Identity();
  } else {
    JIXIE::singularValueDecomposition(A, u, sig, v);
  }

  *reinterpret_cast<Eigen::Matrix2d *>(u_) = u;
  *reinterpret_cast<Eigen::Matrix<double, 2, 1> *>(sig_) = sig;
  *reinterpret_cast<Eigen::Matrix2d *>(v_) = v;
}

#define SPECIALIZE(T, dim)                                                    \
  template void eigen_svd<dim, T>(const MatrixND<dim, T> &,                   \
                                  MatrixND<dim, T> &, MatrixND<dim, T> &,     \
                                  MatrixND<dim, T> &);                        \
  template void imp_svd<dim, T>(const MatrixND<dim, T> &, MatrixND<dim, T> &, \
                                MatrixND<dim, T> &, MatrixND<dim, T> &);      \
  template void svd<dim, T>(const MatrixND<dim, T> &, MatrixND<dim, T> &,     \
                            MatrixND<dim, T> &, MatrixND<dim, T> &);          \
  template void svd_rot<dim, T>(const MatrixND<dim, T> &, MatrixND<dim, T> &, \
                                MatrixND<dim, T> &, MatrixND<dim, T> &);      \
  template void qr_decomp<dim, T>(const MatrixND<dim, T> &,                   \
                                  MatrixND<dim, T> &, MatrixND<dim, T> &T);   \
  template void polar_decomp<dim, T>(const MatrixND<dim, T> &,                \
                                     MatrixND<dim, T> &, MatrixND<dim, T> &T);

SPECIALIZE(float32, 2);
SPECIALIZE(float32, 3);
SPECIALIZE(float64, 2);
SPECIALIZE(float64, 3);

TC_NAMESPACE_END
