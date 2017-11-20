/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include <implicit_qr_svd/Tools.h>
#include <implicit_qr_svd/ImplicitQRSVD.h>
#include <Eigen/Dense>
#pragma GCC diagnostic pop
#include "qr_svd.h"

//#define TC_USE_EIGEN_SVD

TC_NAMESPACE_BEGIN

void eigen_svd(Matrix2 m, Matrix2 &u, Matrix2 &s, Matrix2 &v) {
  Eigen::MatrixXf e_m(2, 2);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      e_m(j, i) = m[i][j];
  Eigen::JacobiSVD<Eigen::MatrixXf> e_svd(
      e_m, Eigen::ComputeThinU | Eigen::ComputeThinV);
  s = Matrix2(0.0_f);
  for (int i = 0; i < 2; ++i)
    s[i][i] = e_svd.singularValues()(i);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      u[j][i] = e_svd.matrixU()(i, j);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      v[j][i] = e_svd.matrixV()(i, j);
}

void eigen_svd(Matrix3 m, Matrix3 &u, Matrix3 &s, Matrix3 &v) {
  Eigen::MatrixXf e_m(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      e_m(j, i) = m[i][j];
  Eigen::JacobiSVD<Eigen::MatrixXf> e_svd(
      e_m, Eigen::ComputeThinU | Eigen::ComputeThinV);
  s = Matrix3(0.0_f);
  for (int i = 0; i < 3; ++i)
    s[i][i] = e_svd.singularValues()(i);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      u[j][i] = e_svd.matrixU()(i, j);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      v[j][i] = e_svd.matrixV()(i, j);
}

void ensure_non_negative_singular_values(Matrix2 &u, Matrix2 &s) {
  if (s[0][0] < 0) {
    s[0][0] = -s[0][0];
    u[0] *= -1.0_f;
  }
  if (s[1][1] < 0) {
    s[1][1] = -s[1][1];
    u[1] *= -1;
  }
}

void ensure_non_negative_singular_values(Matrix3 &u, Matrix3 &s) {
  if (s[0][0] < 0) {
    s[0][0] = -s[0][0];
    u[0] *= -1;
  }
  if (s[1][1] < 0) {
    s[1][1] = -s[1][1];
    u[1] *= -1;
  }
  if (s[2][2] < 0) {
    s[2][2] = -s[2][2];
    u[2] *= -1;
  }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
// m can not be const here, otherwise JIXIE::singularValueDecomposition will
// cause a error due to const_cast
void imp_svd(Matrix2 m, Matrix2 &u, Matrix2 &s, Matrix2 &v) {
  if ((m - Matrix2(Vector2(m[0][0], m[1][1]))).frobenius_norm2() < 1e-7f) {
    s = m;
    u = v = Matrix2(1);
  } else {
    JIXIE::singularValueDecomposition(
        *(Eigen::Matrix<real, 2, 2> *)&m, *(Eigen::Matrix<real, 2, 2> *)&u,
        *(Eigen::Matrix<real, 2, 1> *)&s, *(Eigen::Matrix<real, 2, 2> *)&v);
    real s_tmp[]{s[0][1]};
    memset(&s[0][0] + 1, 0, sizeof(real) * 3);
    s[1][1] = s_tmp[0];
  }
}
#pragma GCC diagnostic pop

void imp_svd(Matrix3 m, Matrix3 &u, Matrix3 &s, Matrix3 &v) {
  Eigen::Matrix<real, 3, 3> M, U, V;
  Eigen::Matrix<real, 3, 1> S;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      M(j, i) = m[i][j];
    }
  }

  JIXIE::singularValueDecomposition(M, U, S, V);
  s = u = v = Matrix3(0.0_f);
  for (int i = 0; i < 3; i++) {
    s[i][i] = S(i, 0);
  }
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      u[i][j] = U(j, i);
      v[i][j] = V(j, i);
    }
  }
}

void svd(Matrix2 m, Matrix2 &u, Matrix2 &sig, Matrix2 &v) {
#ifdef TC_USE_EIGEN_SVD
  eigen_svd(m, u, sig, v);
#else
  imp_svd(m, u, sig, v);
#endif
  ensure_non_negative_singular_values(u, sig);
}

void svd(Matrix3 m, Matrix3 &u, Matrix3 &sig, Matrix3 &v) {
#ifdef TC_USE_EIGEN_SVD
  eigen_svd(m, u, sig, v);
#else
  if ((m - Matrix3(Vector3(m[0][0], m[1][1], m[2][2]))).frobenius_norm2() <
      1e-7_f) {
    // QR_SVD crashes in this case...
    sig = m;
    u = v = Matrix3(1);
  } else {
    imp_svd(m, u, sig, v);
  }
#endif
  ensure_non_negative_singular_values(u, sig);
}

void qr_decomp(Matrix2 A, Matrix2 &q, Matrix2 &r) {
  real a = A[0][0], b = A[0][1], inv_r = 1.0_f / std::hypot(a, b);
  a *= inv_r;
  b *= inv_r;
  Matrix2 Q(Vector2(a, -b), Vector2(b, a));
  q = Q.transposed();
  r = Q * A;
  for (int i = 0; i < 2; i++) {
    if (r[i][i] < 0) {
      r[0][i] *= -1;
      r[1][i] *= -1;
      q[i] *= -1;
    }
  }
}

void qr_decomp(Matrix3 A, Matrix3 &q, Matrix3 &r) {
  TC_NOT_IMPLEMENTED
}

void polar_decomp(Matrix2 A, Matrix2 &r, Matrix2 &s) {
  Matrix2 u, sig, v;
  svd(A, u, sig, v);
  r = u * transposed(v);
  s = v * sig * transposed(v);
}

void polar_decomp(Matrix3 A, Matrix3 &r, Matrix3 &s) {
  Matrix3 u, sig, v;
  svd(A, u, sig, v);
  r = u * transposed(v);
  s = v * sig * transposed(v);
  if (r.abnormal()) {
    Matrix3 m = A;
    svd(m, u, sig, v);
    TC_P(A);
    TC_P(m);
    TC_P(u);
    TC_P(sig);
    TC_P(v);
    TC_P(r);
    TC_P(s);
    TC_P(transposed(v));
    TC_P(u * transposed(v));
    r = u * transposed(v);
    TC_P(r);
    printf(
        "Matrix3 m(%.30f,%.30f,%.30f,%.30f,%.30f,%.30f,%.30f,%.30f,%.30f);\n",
        m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2],
        m[2][2]);
  }
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

TC_NAMESPACE_END
