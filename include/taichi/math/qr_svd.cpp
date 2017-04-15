/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <implicit_qr_svd/Tools.h>
#include <implicit_qr_svd/ImplicitQRSVD.h>
#include "qr_svd.h"

TC_NAMESPACE_BEGIN

void imp_svd(const Matrix3 &m, Matrix3 &u, Matrix3 &s, Matrix3 &v) {
    /*
    Eigen::Matrix<T, 3, 3> *M;
    Eigen::Matrix<T, 3, 1> *S;
    Eigen::Matrix<T, 3, 3> *U;
    Eigen::Matrix<T, 3, 3> *V;
    */
    JIXIE::singularValueDecomposition(
            *(Eigen::Matrix<float, 3, 3> *)&m,
            *(Eigen::Matrix<float, 3, 3> *)&v,
            *(Eigen::Matrix<float, 3, 1> *)&s,
            *(Eigen::Matrix<float, 3, 3> *)&u
    );
    float s_tmp[2]{s[0][1], s[0][2]};
    memset(&s[0][0] + 1, 0, sizeof(float) * 8);
    s[1][1] = s_tmp[0];
    s[2][2] = s_tmp[1];
    /*
    printf("glm\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf(" %.4f", m[j][i]);
        }
        printf("\n");
    }
    printf("\n");
    printf("eigen\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf(" %.4f", (*((Eigen::Matrix<float, 3, 3> *) &m))(i, j));
        }
        printf("\n");
    }
     */
}

TC_NAMESPACE_END
