//#####################################################################
// Copyright (c) 2010-2011, Eftychios Sifakis.
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

//###########################################################
// Local variable declarations
//###########################################################

#ifdef PRINT_DEBUGGING_OUTPUT

#ifdef USE_SSE_IMPLEMENTATION
    float buf[4];
    float A11,A21,A31,A12,A22,A32,A13,A23,A33;
    float S11,S21,S31,S22,S32,S33;
#ifdef COMPUTE_V_AS_QUATERNION
    float QVS,QVVX,QVVY,QVVZ;
#endif
#ifdef COMPUTE_V_AS_MATRIX
    float V11,V21,V31,V12,V22,V32,V13,V23,V33;
#endif
#ifdef COMPUTE_U_AS_QUATERNION
    float QUS,QUVX,QUVY,QUVZ;
#endif
#ifdef COMPUTE_U_AS_MATRIX
    float U11,U21,U31,U12,U22,U32,U13,U23,U33;
#endif
#endif

#ifdef USE_AVX_IMPLEMENTATION
    float buf[8];
    float A11,A21,A31,A12,A22,A32,A13,A23,A33;
    float S11,S21,S31,S22,S32,S33;
#ifdef COMPUTE_V_AS_QUATERNION
    float QVS,QVVX,QVVY,QVVZ;
#endif
#ifdef COMPUTE_V_AS_MATRIX
    float V11,V21,V31,V12,V22,V32,V13,V23,V33;
#endif
#ifdef COMPUTE_U_AS_QUATERNION
    float QUS,QUVX,QUVY,QUVZ;
#endif
#ifdef COMPUTE_U_AS_MATRIX
    float U11,U21,U31,U12,U22,U32,U13,U23,U33;
#endif
#endif

#ifdef USE_AVX512_IMPLEMENTATION
    float buf[16];
    float A11,A21,A31,A12,A22,A32,A13,A23,A33;
    float S11,S21,S31,S22,S32,S33;
#ifdef COMPUTE_V_AS_QUATERNION
    float QVS,QVVX,QVVY,QVVZ;
#endif
#ifdef COMPUTE_V_AS_MATRIX
    float V11,V21,V31,V12,V22,V32,V13,V23,V33;
#endif
#ifdef COMPUTE_U_AS_QUATERNION
    float QUS,QUVX,QUVY,QUVZ;
#endif
#ifdef COMPUTE_U_AS_MATRIX
    float U11,U21,U31,U12,U22,U32,U13,U23,U33;
#endif
#endif

#endif

const float Four_Gamma_Squared=sqrt(8.)+3.;
const float Sine_Pi_Over_Eight=.5*sqrt(2.-sqrt(2.));
const float Cosine_Pi_Over_Eight=.5*sqrt(2.+sqrt(2.));
const unsigned One_Mask=0xffffffff;

ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sfour_gamma_squared;)       ENABLE_SSE_IMPLEMENTATION(__m128 Vfour_gamma_squared;)                                    ENABLE_AVX_IMPLEMENTATION(__m256 Vfour_gamma_squared;)                                    ENABLE_AVX512_IMPLEMENTATION(__m512 Vfour_gamma_squared;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Ssine_pi_over_eight;)       ENABLE_SSE_IMPLEMENTATION(__m128 Vsine_pi_over_eight;)                                    ENABLE_AVX_IMPLEMENTATION(__m256 Vsine_pi_over_eight;)                                    ENABLE_AVX512_IMPLEMENTATION(__m512 Vsine_pi_over_eight;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Scosine_pi_over_eight;)     ENABLE_SSE_IMPLEMENTATION(__m128 Vcosine_pi_over_eight;)                                  ENABLE_AVX_IMPLEMENTATION(__m256 Vcosine_pi_over_eight;)                                  ENABLE_AVX512_IMPLEMENTATION(__m512 Vcosine_pi_over_eight;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sone_half;)                 ENABLE_SSE_IMPLEMENTATION(__m128 Vone_half;)                                              ENABLE_AVX_IMPLEMENTATION(__m256 Vone_half;)                                              ENABLE_AVX512_IMPLEMENTATION(__m512 Vone_half;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sone;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vone;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vone;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vone;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Stiny_number;)              ENABLE_SSE_IMPLEMENTATION(__m128 Vtiny_number;)                                           ENABLE_AVX_IMPLEMENTATION(__m256 Vtiny_number;)                                           ENABLE_AVX512_IMPLEMENTATION(__m512 Vtiny_number;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Ssmall_number;)             ENABLE_SSE_IMPLEMENTATION(__m128 Vsmall_number;)                                          ENABLE_AVX_IMPLEMENTATION(__m256 Vsmall_number;)                                          ENABLE_AVX512_IMPLEMENTATION(__m512 Vsmall_number;)
                                                                                                                                                                                                                                                                              ENABLE_AVX512_IMPLEMENTATION(__m512 Vone_mask;)

ENABLE_SCALAR_IMPLEMENTATION(Sfour_gamma_squared.f=Four_Gamma_Squared;)                   ENABLE_SSE_IMPLEMENTATION(Vfour_gamma_squared=_mm_set1_ps(Four_Gamma_Squared);)           ENABLE_AVX_IMPLEMENTATION(Vfour_gamma_squared=_mm256_set1_ps(Four_Gamma_Squared);)        ENABLE_AVX512_IMPLEMENTATION(Vfour_gamma_squared=_mm512_set1_ps(Four_Gamma_Squared);)
ENABLE_SCALAR_IMPLEMENTATION(Ssine_pi_over_eight.f=Sine_Pi_Over_Eight;)                   ENABLE_SSE_IMPLEMENTATION(Vsine_pi_over_eight=_mm_set1_ps(Sine_Pi_Over_Eight);)           ENABLE_AVX_IMPLEMENTATION(Vsine_pi_over_eight=_mm256_set1_ps(Sine_Pi_Over_Eight);)        ENABLE_AVX512_IMPLEMENTATION(Vsine_pi_over_eight=_mm512_set1_ps(Sine_Pi_Over_Eight);)
ENABLE_SCALAR_IMPLEMENTATION(Scosine_pi_over_eight.f=Cosine_Pi_Over_Eight;)               ENABLE_SSE_IMPLEMENTATION(Vcosine_pi_over_eight=_mm_set1_ps(Cosine_Pi_Over_Eight);)       ENABLE_AVX_IMPLEMENTATION(Vcosine_pi_over_eight=_mm256_set1_ps(Cosine_Pi_Over_Eight);)    ENABLE_AVX512_IMPLEMENTATION(Vcosine_pi_over_eight=_mm512_set1_ps(Cosine_Pi_Over_Eight);)
ENABLE_SCALAR_IMPLEMENTATION(Sone_half.f=.5;)                                             ENABLE_SSE_IMPLEMENTATION(Vone_half=_mm_set1_ps(.5);)                                     ENABLE_AVX_IMPLEMENTATION(Vone_half=_mm256_set1_ps(.5);)                                  ENABLE_AVX512_IMPLEMENTATION(Vone_half=_mm512_set1_ps(.5);)
ENABLE_SCALAR_IMPLEMENTATION(Sone.f=1.;)                                                  ENABLE_SSE_IMPLEMENTATION(Vone=_mm_set1_ps(1.);)                                          ENABLE_AVX_IMPLEMENTATION(Vone=_mm256_set1_ps(1.);)                                       ENABLE_AVX512_IMPLEMENTATION(Vone=_mm512_set1_ps(1.);)
ENABLE_SCALAR_IMPLEMENTATION(Stiny_number.f=1.e-20;)                                      ENABLE_SSE_IMPLEMENTATION(Vtiny_number=_mm_set1_ps(1.e-20);)                              ENABLE_AVX_IMPLEMENTATION(Vtiny_number=_mm256_set1_ps(1.e-20);)                           ENABLE_AVX512_IMPLEMENTATION(Vtiny_number=_mm512_set1_ps(1.e-20);)
ENABLE_SCALAR_IMPLEMENTATION(Ssmall_number.f=1.e-12;)                                     ENABLE_SSE_IMPLEMENTATION(Vsmall_number=_mm_set1_ps(1.e-12);)                             ENABLE_AVX_IMPLEMENTATION(Vsmall_number=_mm256_set1_ps(1.e-12);)                          ENABLE_AVX512_IMPLEMENTATION(Vsmall_number=_mm512_set1_ps(1.e-12);)
                                                                                                                                                                                                                                                                              ENABLE_AVX512_IMPLEMENTATION(Vone_mask=_mm512_castsi512_ps(_mm512_set1_epi32(One_Mask));)

ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sa11;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Va11;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Va11;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Va11;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sa21;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Va21;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Va21;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Va21;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sa31;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Va31;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Va31;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Va31;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sa12;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Va12;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Va12;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Va12;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sa22;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Va22;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Va22;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Va22;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sa32;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Va32;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Va32;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Va32;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sa13;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Va13;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Va13;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Va13;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sa23;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Va23;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Va23;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Va23;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sa33;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Va33;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Va33;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Va33;)

#ifdef COMPUTE_V_AS_MATRIX
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sv11;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vv11;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vv11;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vv11;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sv21;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vv21;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vv21;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vv21;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sv31;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vv31;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vv31;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vv31;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sv12;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vv12;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vv12;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vv12;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sv22;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vv22;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vv22;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vv22;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sv32;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vv32;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vv32;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vv32;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sv13;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vv13;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vv13;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vv13;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sv23;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vv23;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vv23;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vv23;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sv33;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vv33;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vv33;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vv33;)
#endif

#ifdef COMPUTE_V_AS_QUATERNION
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sqvs;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vqvs;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vqvs;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vqvs;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sqvvx;)                     ENABLE_SSE_IMPLEMENTATION(__m128 Vqvvx;)                                                  ENABLE_AVX_IMPLEMENTATION(__m256 Vqvvx;)                                                  ENABLE_AVX512_IMPLEMENTATION(__m512 Vqvvx;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sqvvy;)                     ENABLE_SSE_IMPLEMENTATION(__m128 Vqvvy;)                                                  ENABLE_AVX_IMPLEMENTATION(__m256 Vqvvy;)                                                  ENABLE_AVX512_IMPLEMENTATION(__m512 Vqvvy;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sqvvz;)                     ENABLE_SSE_IMPLEMENTATION(__m128 Vqvvz;)                                                  ENABLE_AVX_IMPLEMENTATION(__m256 Vqvvz;)                                                  ENABLE_AVX512_IMPLEMENTATION(__m512 Vqvvz;)
#endif

#ifdef COMPUTE_U_AS_MATRIX
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Su11;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vu11;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vu11;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vu11;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Su21;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vu21;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vu21;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vu21;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Su31;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vu31;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vu31;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vu31;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Su12;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vu12;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vu12;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vu12;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Su22;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vu22;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vu22;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vu22;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Su32;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vu32;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vu32;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vu32;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Su13;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vu13;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vu13;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vu13;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Su23;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vu23;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vu23;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vu23;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Su33;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vu33;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vu33;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vu33;)
#endif

#ifdef COMPUTE_U_AS_QUATERNION
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Squs;)                      ENABLE_SSE_IMPLEMENTATION(__m128 Vqus;)                                                   ENABLE_AVX_IMPLEMENTATION(__m256 Vqus;)                                                   ENABLE_AVX512_IMPLEMENTATION(__m512 Vqus;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Squvx;)                     ENABLE_SSE_IMPLEMENTATION(__m128 Vquvx;)                                                  ENABLE_AVX_IMPLEMENTATION(__m256 Vquvx;)                                                  ENABLE_AVX512_IMPLEMENTATION(__m512 Vquvx;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Squvy;)                     ENABLE_SSE_IMPLEMENTATION(__m128 Vquvy;)                                                  ENABLE_AVX_IMPLEMENTATION(__m256 Vquvy;)                                                  ENABLE_AVX512_IMPLEMENTATION(__m512 Vquvy;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Squvz;)                     ENABLE_SSE_IMPLEMENTATION(__m128 Vquvz;)                                                  ENABLE_AVX_IMPLEMENTATION(__m256 Vquvz;)                                                  ENABLE_AVX512_IMPLEMENTATION(__m512 Vquvz;)
#endif

ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sc;)                        ENABLE_SSE_IMPLEMENTATION(__m128 Vc;)                                                     ENABLE_AVX_IMPLEMENTATION(__m256 Vc;)                                                     ENABLE_AVX512_IMPLEMENTATION(__m512 Vc;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Ss;)                        ENABLE_SSE_IMPLEMENTATION(__m128 Vs;)                                                     ENABLE_AVX_IMPLEMENTATION(__m256 Vs;)                                                     ENABLE_AVX512_IMPLEMENTATION(__m512 Vs;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Sch;)                       ENABLE_SSE_IMPLEMENTATION(__m128 Vch;)                                                    ENABLE_AVX_IMPLEMENTATION(__m256 Vch;)                                                    ENABLE_AVX512_IMPLEMENTATION(__m512 Vch;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Ssh;)                       ENABLE_SSE_IMPLEMENTATION(__m128 Vsh;)                                                    ENABLE_AVX_IMPLEMENTATION(__m256 Vsh;)                                                    ENABLE_AVX512_IMPLEMENTATION(__m512 Vsh;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Stmp1;)                     ENABLE_SSE_IMPLEMENTATION(__m128 Vtmp1;)                                                  ENABLE_AVX_IMPLEMENTATION(__m256 Vtmp1;)                                                  ENABLE_AVX512_IMPLEMENTATION(__m512 Vtmp1;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Stmp2;)                     ENABLE_SSE_IMPLEMENTATION(__m128 Vtmp2;)                                                  ENABLE_AVX_IMPLEMENTATION(__m256 Vtmp2;)                                                  ENABLE_AVX512_IMPLEMENTATION(__m512 Vtmp2;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Stmp3;)                     ENABLE_SSE_IMPLEMENTATION(__m128 Vtmp3;)                                                  ENABLE_AVX_IMPLEMENTATION(__m256 Vtmp3;)                                                  ENABLE_AVX512_IMPLEMENTATION(__m512 Vtmp3;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Stmp4;)                     ENABLE_SSE_IMPLEMENTATION(__m128 Vtmp4;)                                                  ENABLE_AVX_IMPLEMENTATION(__m256 Vtmp4;)                                                  ENABLE_AVX512_IMPLEMENTATION(__m512 Vtmp4;)
ENABLE_SCALAR_IMPLEMENTATION(union {float f;unsigned int ui;} Stmp5;)                     ENABLE_SSE_IMPLEMENTATION(__m128 Vtmp5;)                                                  ENABLE_AVX_IMPLEMENTATION(__m256 Vtmp5;)                                                  ENABLE_AVX512_IMPLEMENTATION(__m512 Vtmp5;)
                                                                                                                                                                                                                                                                              ENABLE_AVX512_IMPLEMENTATION(__mmask16 Mtmp;)
