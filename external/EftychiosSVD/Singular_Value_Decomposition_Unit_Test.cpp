//#####################################################################
// Copyright (c) 2009-2011, Eftychios Sifakis.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//   * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
//     other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
// BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//#####################################################################

#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#define PRINT_DEBUGGING_OUTPUT
#define USE_SCALAR_IMPLEMENTATION
#define USE_ACCURATE_RSQRT_IN_JACOBI_CONJUGATION
// #define USE_SSE_IMPLEMENTATION
// #define USE_AVX_IMPLEMENTATION

#define COMPUTE_V_AS_MATRIX
#define COMPUTE_V_AS_QUATERNION
#define COMPUTE_U_AS_MATRIX
#define COMPUTE_U_AS_QUATERNION

#include "Singular_Value_Decomposition_Preamble.hpp"

int main(int argc,char* argv[])
{
    if(argc!=2){printf("Must specify an integer random seed as argument\n");exit(1);}
    srand(atoi(argv[1]));

    float test_A11,test_A21,test_A31,test_A12,test_A22,test_A32,test_A13,test_A23,test_A33;

    test_A11=2.*(float)rand()/(float)RAND_MAX-1.;
    test_A21=2.*(float)rand()/(float)RAND_MAX-1.;
    test_A31=2.*(float)rand()/(float)RAND_MAX-1.;
    test_A12=2.*(float)rand()/(float)RAND_MAX-1.;
    test_A22=2.*(float)rand()/(float)RAND_MAX-1.;
    test_A32=2.*(float)rand()/(float)RAND_MAX-1.;
    test_A13=2.*(float)rand()/(float)RAND_MAX-1.;
    test_A23=2.*(float)rand()/(float)RAND_MAX-1.;
    test_A33=2.*(float)rand()/(float)RAND_MAX-1.;
    float norm_inverse=(float)(1./sqrt((double)test_A11*(double)test_A11+(double)test_A21*(double)test_A21+(double)test_A31*(double)test_A31
            +(double)test_A12*(double)test_A12+(double)test_A22*(double)test_A22+(double)test_A32*(double)test_A32
            +(double)test_A13*(double)test_A13+(double)test_A23*(double)test_A23+(double)test_A33*(double)test_A33));
    test_A11*=norm_inverse;
    test_A21*=norm_inverse;
    test_A31*=norm_inverse;
    test_A12*=norm_inverse;
    test_A22*=norm_inverse;
    test_A32*=norm_inverse;
    test_A13*=norm_inverse;
    test_A23*=norm_inverse;
    test_A33*=norm_inverse;
        
#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"

    ENABLE_SCALAR_IMPLEMENTATION(Sa11.f=test_A11;)                                        ENABLE_SSE_IMPLEMENTATION(Va11=_mm_set1_ps(test_A11);)                                    ENABLE_AVX_IMPLEMENTATION(Va11=_mm256_set1_ps(test_A11);)                                 ENABLE_AVX512_IMPLEMENTATION(Va11=_mm512_set1_ps(test_A11);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa21.f=test_A21;)                                        ENABLE_SSE_IMPLEMENTATION(Va21=_mm_set1_ps(test_A21);)                                    ENABLE_AVX_IMPLEMENTATION(Va21=_mm256_set1_ps(test_A21);)                                 ENABLE_AVX512_IMPLEMENTATION(Va21=_mm512_set1_ps(test_A21);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa31.f=test_A31;)                                        ENABLE_SSE_IMPLEMENTATION(Va31=_mm_set1_ps(test_A31);)                                    ENABLE_AVX_IMPLEMENTATION(Va31=_mm256_set1_ps(test_A31);)                                 ENABLE_AVX512_IMPLEMENTATION(Va31=_mm512_set1_ps(test_A31);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa12.f=test_A12;)                                        ENABLE_SSE_IMPLEMENTATION(Va12=_mm_set1_ps(test_A12);)                                    ENABLE_AVX_IMPLEMENTATION(Va12=_mm256_set1_ps(test_A12);)                                 ENABLE_AVX512_IMPLEMENTATION(Va12=_mm512_set1_ps(test_A12);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa22.f=test_A22;)                                        ENABLE_SSE_IMPLEMENTATION(Va22=_mm_set1_ps(test_A22);)                                    ENABLE_AVX_IMPLEMENTATION(Va22=_mm256_set1_ps(test_A22);)                                 ENABLE_AVX512_IMPLEMENTATION(Va22=_mm512_set1_ps(test_A22);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa32.f=test_A32;)                                        ENABLE_SSE_IMPLEMENTATION(Va32=_mm_set1_ps(test_A32);)                                    ENABLE_AVX_IMPLEMENTATION(Va32=_mm256_set1_ps(test_A32);)                                 ENABLE_AVX512_IMPLEMENTATION(Va32=_mm512_set1_ps(test_A32);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa13.f=test_A13;)                                        ENABLE_SSE_IMPLEMENTATION(Va13=_mm_set1_ps(test_A13);)                                    ENABLE_AVX_IMPLEMENTATION(Va13=_mm256_set1_ps(test_A13);)                                 ENABLE_AVX512_IMPLEMENTATION(Va13=_mm512_set1_ps(test_A13);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa23.f=test_A23;)                                        ENABLE_SSE_IMPLEMENTATION(Va23=_mm_set1_ps(test_A23);)                                    ENABLE_AVX_IMPLEMENTATION(Va23=_mm256_set1_ps(test_A23);)                                 ENABLE_AVX512_IMPLEMENTATION(Va23=_mm512_set1_ps(test_A23);)
    ENABLE_SCALAR_IMPLEMENTATION(Sa33.f=test_A33;)                                        ENABLE_SSE_IMPLEMENTATION(Va33=_mm_set1_ps(test_A33);)                                    ENABLE_AVX_IMPLEMENTATION(Va33=_mm256_set1_ps(test_A33);)                                 ENABLE_AVX512_IMPLEMENTATION(Va33=_mm512_set1_ps(test_A33);)

#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp"    

    return 0;
}
//#####################################################################
