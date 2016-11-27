#include "Tools.h"
#include "ImplicitQRSVD.h"
#include "qr_svd.h"

#include <mmintrin.h>
#include <xmmintrin.h>
TC_NAMESPACE_BEGIN

void imp_svd(const Matrix3 &m, Matrix3 &u, Matrix3 &s, Matrix3 &v) {
	/*
	Eigen::Matrix<T, 3, 3> *M;
	Eigen::Matrix<T, 3, 1> *S;
	Eigen::Matrix<T, 3, 3> *U;
	Eigen::Matrix<T, 3, 3> *V;
	*/

	 printf(">_< %f\n", _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(0.00000000000000000000000000000000000000000932704258f))));


	real A[] = {
	0.999999880790710449218750000000f, 0.000000000000014210854715202004f, -0.000000000000003552713678800501f,
	-0.000000000000014210854715202004f, 0.999999821186065673828125000000f, -0.000000000000001744644772887997f,
	0.000000000000003552713678800501f, 0.000000119209289550781250000000f, 1.000000000000000000000000000000f
	};
        JIXIE::singularValueDecomposition(
                //*(Eigen::Matrix<float, 3, 3> *) &m,
                *(Eigen::Matrix<float, 3, 3> *) A,
                *(Eigen::Matrix<float, 3, 3> *) &v,
                *(Eigen::Matrix<float, 3, 1> *) &s,
                *(Eigen::Matrix<float, 3, 3> *) &u
        );
        float s_tmp[2]{s[0][1], s[0][2]};
        memset(&s[0][0] + 1, 0, sizeof(real) * 8);
        s[1][1] = s_tmp[0];
        s[2][2] = s_tmp[1];
        printf("glm\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf(" %.30f", m[j][i]);
            }
            printf("\n");
        }
        printf("\n");
        printf("eigen\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf(" %.10f", (*((Eigen::Matrix<float, 3, 3> *) &m))(i, j));
            }
            printf("\n");
        }
    }

TC_NAMESPACE_END
