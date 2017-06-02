/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <implicit_qr_svd/Tools.h>
#include <implicit_qr_svd/ImplicitQRSVD.h>
#include <Eigen/Dense>
#include "qr_svd.h"

#define TC_USE_EIGEN_SVD

TC_NAMESPACE_BEGIN


void eigen_svd(Matrix2 m, Matrix2 &u, Matrix2 &s, Matrix2 &v) {
    Eigen::MatrixXf e_m(2, 2);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            e_m(j, i) = m[i][j];
    Eigen::JacobiSVD<Eigen::MatrixXf> e_svd(e_m, Eigen::ComputeThinU | Eigen::ComputeThinV);
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
    Eigen::JacobiSVD<Eigen::MatrixXf> e_svd(e_m, Eigen::ComputeThinU | Eigen::ComputeThinV);
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
        u[0][0] *= -1;
        u[0][1] *= -1;
    }
    if (s[1][1] < 0) {
        s[1][1] = -s[1][1];
        u[1][0] *= -1;
        u[1][1] *= -1;
    }
}

void ensure_non_negative_singular_values(Matrix3 &u, Matrix3 &s) {
    if (s[0][0] < 0) {
        s[0][0] = -s[0][0];
        u[0][0] *= -1;
        u[0][1] *= -1;
        u[0][2] *= -1;
    }
    if (s[1][1] < 0) {
        s[1][1] = -s[1][1];
        u[1][0] *= -1;
        u[1][1] *= -1;
        u[1][2] *= -1;
    }
    if (s[2][2] < 0) {
        s[2][2] = -s[2][2];
        u[2][0] *= -1;
        u[2][1] *= -1;
        u[2][2] *= -1;
    }
}


// m can not be const here, otherwise JIXIE::singularValueDecomposition will cause a error due to const_cast
void imp_svd(Matrix2 m, Matrix2 &u, Matrix2 &s, Matrix2 &v) {
    if (frobenius_norm2(m - Matrix2(m[0][0], 0, 0, m[1][1])) < 1e-7f) {
        s = m;
        u = v = Matrix2(1);
    } else {
        JIXIE::singularValueDecomposition(
                *(Eigen::Matrix<float, 2, 2> *)&m,
                *(Eigen::Matrix<float, 2, 2> *)&u,
                *(Eigen::Matrix<float, 2, 1> *)&s,
                *(Eigen::Matrix<float, 2, 2> *)&v
        );
        float s_tmp[]{s[0][1]};
        memset(&s[0][0] + 1, 0, sizeof(float) * 3);
        s[1][1] = s_tmp[0];
    }
}

void imp_svd(Matrix3 m, Matrix3 &u, Matrix3 &s, Matrix3 &v) {
    JIXIE::singularValueDecomposition(
            *(Eigen::Matrix<float, 3, 3> *)&m,
            *(Eigen::Matrix<float, 3, 3> *)&u,
            *(Eigen::Matrix<float, 3, 1> *)&s,
            *(Eigen::Matrix<float, 3, 3> *)&v
    );
    float s_tmp[2]{s[0][1], s[0][2]};
    memset(&s[0][0] + 1, 0, sizeof(float) * 8);
    s[1][1] = s_tmp[0];
    s[2][2] = s_tmp[1];
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
    if (frobenius_norm2(m - Matrix3(m[0][0], 0, 0, 0, m[1][1], 0, 0, 0, m[2][2])) < 1e-7f) {
        // QR_SVD crashes in this case...
        sig = m;
        u = v = Matrix3(1);
    } else {
        imp_svd(m, u, sig, v);
    }
#endif
    ensure_non_negative_singular_values(u, sig);
}


void polar_decomp(Matrix2 A, Matrix2 &r, Matrix2 &s) {
    Matrix2 u, sig, v;
    svd(A, u, sig, v);
    r = u * glm::transpose(v);
    s = v * sig * glm::transpose(v);
}

void polar_decomp(Matrix3 A, Matrix3 &r, Matrix3 &s) {
    Matrix3 u, sig, v;
    svd(A, u, sig, v);
    r = u * glm::transpose(v);
    s = v * sig * glm::transpose(v);
    if (!is_normal(r)) {
        Matrix3 m = A;
        svd(m, u, sig, v);
        P(A);
        P(m);
        P(u);
        P(sig);
        P(v);
        P(r);
        P(s);
        P(glm::transpose(v));
        P(u * glm::transpose(v));
        r = u * glm::transpose(v);
        P(r);
        printf("Matrix3 m(%.30f,%.30f,%.30f,%.30f,%.30f,%.30f,%.30f,%.30f,%.30f);\n", m[0][0], m[1][0], m[2][0],
               m[0][1],
               m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]);
    }
}

TC_NAMESPACE_END
