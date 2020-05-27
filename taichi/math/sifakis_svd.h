#include <cmath>
#include <algorithm>
#include "taichi/common/core.h"

namespace SifakisSVD {

/*
http://pages.cs.wisc.edu/~sifakis/project_pages/svd.html

Computing the Singular Value Decomposition of 3x3 matrices with minimal
branching and elementary floating point operations

A. McAdams, A. Selle, R. Tamstorf, J. Teran and E. Sifakis
*/

//#####################################################################
// Copyright (c) 2010-2011, Eftychios Sifakis.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//   * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or
//     other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//#####################################################################

TI_FORCE_INLINE float rsqrt(const float f) {
  return 1.0f / std::sqrt(f);
}

constexpr float Four_Gamma_Squared = 5.82842712474619f;  // sqrt(8.) + 3.;
constexpr float Sine_Pi_Over_Eight =
    0.3826834323650897f;  // .5 * sqrt(2. - sqrt(2.));
constexpr float Cosine_Pi_Over_Eight =
    0.9238795325112867f;  //.5 * sqrt(2. + sqrt(2.));

template <int sweeps = 4>
TI_FORCE_INLINE void svd(const float a11,
                         const float a12,
                         const float a13,
                         const float a21,
                         const float a22,
                         const float a23,
                         const float a31,
                         const float a32,
                         const float a33,
                         float &u11,
                         float &u12,
                         float &u13,
                         float &u21,
                         float &u22,
                         float &u23,
                         float &u31,
                         float &u32,
                         float &u33,
                         float &v11,
                         float &v12,
                         float &v13,
                         float &v21,
                         float &v22,
                         float &v23,
                         float &v31,
                         float &v32,
                         float &v33,
                         float &sigma1,
                         float &sigma2,
                         float &sigma3) {
  // var
  union {
    float f;
    unsigned int ui;
  } Sfour_gamma_squared;
  union {
    float f;
    unsigned int ui;
  } Ssine_pi_over_eight;
  union {
    float f;
    unsigned int ui;
  } Scosine_pi_over_eight;
  union {
    float f;
    unsigned int ui;
  } Sone_half;
  union {
    float f;
    unsigned int ui;
  } Sone;
  union {
    float f;
    unsigned int ui;
  } Stiny_number;
  union {
    float f;
    unsigned int ui;
  } Ssmall_number;
  union {
    float f;
    unsigned int ui;
  } Sa11;
  union {
    float f;
    unsigned int ui;
  } Sa21;
  union {
    float f;
    unsigned int ui;
  } Sa31;
  union {
    float f;
    unsigned int ui;
  } Sa12;
  union {
    float f;
    unsigned int ui;
  } Sa22;
  union {
    float f;
    unsigned int ui;
  } Sa32;
  union {
    float f;
    unsigned int ui;
  } Sa13;
  union {
    float f;
    unsigned int ui;
  } Sa23;
  union {
    float f;
    unsigned int ui;
  } Sa33;

  union {
    float f;
    unsigned int ui;
  } Sv11;
  union {
    float f;
    unsigned int ui;
  } Sv21;
  union {
    float f;
    unsigned int ui;
  } Sv31;
  union {
    float f;
    unsigned int ui;
  } Sv12;
  union {
    float f;
    unsigned int ui;
  } Sv22;
  union {
    float f;
    unsigned int ui;
  } Sv32;
  union {
    float f;
    unsigned int ui;
  } Sv13;
  union {
    float f;
    unsigned int ui;
  } Sv23;
  union {
    float f;
    unsigned int ui;
  } Sv33;
  union {
    float f;
    unsigned int ui;
  } Su11;
  union {
    float f;
    unsigned int ui;
  } Su21;
  union {
    float f;
    unsigned int ui;
  } Su31;
  union {
    float f;
    unsigned int ui;
  } Su12;
  union {
    float f;
    unsigned int ui;
  } Su22;
  union {
    float f;
    unsigned int ui;
  } Su32;
  union {
    float f;
    unsigned int ui;
  } Su13;
  union {
    float f;
    unsigned int ui;
  } Su23;
  union {
    float f;
    unsigned int ui;
  } Su33;
  union {
    float f;
    unsigned int ui;
  } Sc;
  union {
    float f;
    unsigned int ui;
  } Ss;
  union {
    float f;
    unsigned int ui;
  } Sch;
  union {
    float f;
    unsigned int ui;
  } Ssh;
  union {
    float f;
    unsigned int ui;
  } Stmp1;
  union {
    float f;
    unsigned int ui;
  } Stmp2;
  union {
    float f;
    unsigned int ui;
  } Stmp3;
  union {
    float f;
    unsigned int ui;
  } Stmp4;
  union {
    float f;
    unsigned int ui;
  } Stmp5;
  union {
    float f;
    unsigned int ui;
  } Sqvs;
  union {
    float f;
    unsigned int ui;
  } Sqvvx;
  union {
    float f;
    unsigned int ui;
  } Sqvvy;
  union {
    float f;
    unsigned int ui;
  } Sqvvz;

  union {
    float f;
    unsigned int ui;
  } Ss11;
  union {
    float f;
    unsigned int ui;
  } Ss21;
  union {
    float f;
    unsigned int ui;
  } Ss31;
  union {
    float f;
    unsigned int ui;
  } Ss22;
  union {
    float f;
    unsigned int ui;
  } Ss32;
  union {
    float f;
    unsigned int ui;
  } Ss33;

  // compute
  Sfour_gamma_squared.f = Four_Gamma_Squared;
  Ssine_pi_over_eight.f = Sine_Pi_Over_Eight;
  Scosine_pi_over_eight.f = Cosine_Pi_Over_Eight;
  Sone_half.f = 0.5f;
  Sone.f = 1.0f;
  Stiny_number.f = 1.e-20f;
  Ssmall_number.f = 1.e-12f;

  Sa11.f = a11;
  Sa21.f = a21;
  Sa31.f = a31;
  Sa12.f = a12;
  Sa22.f = a22;
  Sa32.f = a32;
  Sa13.f = a13;
  Sa23.f = a23;
  Sa33.f = a33;

  Sqvs.f = 1.0f;
  Sqvvx.f = 0.0f;
  Sqvvy.f = 0.0f;
  Sqvvz.f = 0.0f;

  Ss11.f = Sa11.f * Sa11.f;
  Stmp1.f = Sa21.f * Sa21.f;
  Ss11.f = Stmp1.f + Ss11.f;
  Stmp1.f = Sa31.f * Sa31.f;
  Ss11.f = Stmp1.f + Ss11.f;

  Ss21.f = Sa12.f * Sa11.f;
  Stmp1.f = Sa22.f * Sa21.f;
  Ss21.f = Stmp1.f + Ss21.f;
  Stmp1.f = Sa32.f * Sa31.f;
  Ss21.f = Stmp1.f + Ss21.f;

  Ss31.f = Sa13.f * Sa11.f;
  Stmp1.f = Sa23.f * Sa21.f;
  Ss31.f = Stmp1.f + Ss31.f;
  Stmp1.f = Sa33.f * Sa31.f;
  Ss31.f = Stmp1.f + Ss31.f;

  Ss22.f = Sa12.f * Sa12.f;
  Stmp1.f = Sa22.f * Sa22.f;
  Ss22.f = Stmp1.f + Ss22.f;
  Stmp1.f = Sa32.f * Sa32.f;
  Ss22.f = Stmp1.f + Ss22.f;

  Ss32.f = Sa13.f * Sa12.f;
  Stmp1.f = Sa23.f * Sa22.f;
  Ss32.f = Stmp1.f + Ss32.f;
  Stmp1.f = Sa33.f * Sa32.f;
  Ss32.f = Stmp1.f + Ss32.f;

  Ss33.f = Sa13.f * Sa13.f;
  Stmp1.f = Sa23.f * Sa23.f;
  Ss33.f = Stmp1.f + Ss33.f;
  Stmp1.f = Sa33.f * Sa33.f;
  Ss33.f = Stmp1.f + Ss33.f;

  for (int sweep = 0; sweep < sweeps; sweep++) {
    Ssh.f = Ss21.f * Sone_half.f;
    Stmp5.f = Ss11.f - Ss22.f;

    Stmp2.f = Ssh.f * Ssh.f;
    Stmp1.ui = (Stmp2.f >= Stiny_number.f) ? 0xffffffff : 0;
    Ssh.ui = Stmp1.ui & Ssh.ui;
    Sch.ui = Stmp1.ui & Stmp5.ui;
    Stmp2.ui = ~Stmp1.ui & Sone.ui;
    Sch.ui = Sch.ui | Stmp2.ui;

    Stmp1.f = Ssh.f * Ssh.f;
    Stmp2.f = Sch.f * Sch.f;
    Stmp3.f = Stmp1.f + Stmp2.f;
    Stmp4.f = rsqrt(Stmp3.f);
    Ssh.f = Stmp4.f * Ssh.f;
    Sch.f = Stmp4.f * Sch.f;

    Stmp1.f = Sfour_gamma_squared.f * Stmp1.f;
    Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

    Stmp2.ui = Ssine_pi_over_eight.ui & Stmp1.ui;
    Ssh.ui = ~Stmp1.ui & Ssh.ui;
    Ssh.ui = Ssh.ui | Stmp2.ui;
    Stmp2.ui = Scosine_pi_over_eight.ui & Stmp1.ui;
    Sch.ui = ~Stmp1.ui & Sch.ui;
    Sch.ui = Sch.ui | Stmp2.ui;

    Stmp1.f = Ssh.f * Ssh.f;
    Stmp2.f = Sch.f * Sch.f;
    Sc.f = Stmp2.f - Stmp1.f;
    Ss.f = Sch.f * Ssh.f;
    Ss.f = Ss.f + Ss.f;

    Stmp3.f = Stmp1.f + Stmp2.f;
    Ss33.f = Ss33.f * Stmp3.f;
    Ss31.f = Ss31.f * Stmp3.f;
    Ss32.f = Ss32.f * Stmp3.f;
    Ss33.f = Ss33.f * Stmp3.f;

    Stmp1.f = Ss.f * Ss31.f;
    Stmp2.f = Ss.f * Ss32.f;
    Ss31.f = Sc.f * Ss31.f;
    Ss32.f = Sc.f * Ss32.f;
    Ss31.f = Stmp2.f + Ss31.f;
    Ss32.f = Ss32.f - Stmp1.f;

    Stmp2.f = Ss.f * Ss.f;
    Stmp1.f = Ss22.f * Stmp2.f;
    Stmp3.f = Ss11.f * Stmp2.f;
    Stmp4.f = Sc.f * Sc.f;
    Ss11.f = Ss11.f * Stmp4.f;
    Ss22.f = Ss22.f * Stmp4.f;
    Ss11.f = Ss11.f + Stmp1.f;
    Ss22.f = Ss22.f + Stmp3.f;
    Stmp4.f = Stmp4.f - Stmp2.f;
    Stmp2.f = Ss21.f + Ss21.f;
    Ss21.f = Ss21.f * Stmp4.f;
    Stmp4.f = Sc.f * Ss.f;
    Stmp2.f = Stmp2.f * Stmp4.f;
    Stmp5.f = Stmp5.f * Stmp4.f;
    Ss11.f = Ss11.f + Stmp2.f;
    Ss21.f = Ss21.f - Stmp5.f;
    Ss22.f = Ss22.f - Stmp2.f;

    Stmp1.f = Ssh.f * Sqvvx.f;
    Stmp2.f = Ssh.f * Sqvvy.f;
    Stmp3.f = Ssh.f * Sqvvz.f;
    Ssh.f = Ssh.f * Sqvs.f;

    Sqvs.f = Sch.f * Sqvs.f;
    Sqvvx.f = Sch.f * Sqvvx.f;
    Sqvvy.f = Sch.f * Sqvvy.f;
    Sqvvz.f = Sch.f * Sqvvz.f;

    Sqvvz.f = Sqvvz.f + Ssh.f;
    Sqvs.f = Sqvs.f - Stmp3.f;
    Sqvvx.f = Sqvvx.f + Stmp2.f;
    Sqvvy.f = Sqvvy.f - Stmp1.f;
    Ssh.f = Ss32.f * Sone_half.f;
    Stmp5.f = Ss22.f - Ss33.f;

    Stmp2.f = Ssh.f * Ssh.f;
    Stmp1.ui = (Stmp2.f >= Stiny_number.f) ? 0xffffffff : 0;
    Ssh.ui = Stmp1.ui & Ssh.ui;
    Sch.ui = Stmp1.ui & Stmp5.ui;
    Stmp2.ui = ~Stmp1.ui & Sone.ui;
    Sch.ui = Sch.ui | Stmp2.ui;

    Stmp1.f = Ssh.f * Ssh.f;
    Stmp2.f = Sch.f * Sch.f;
    Stmp3.f = Stmp1.f + Stmp2.f;
    Stmp4.f = rsqrt(Stmp3.f);
    Ssh.f = Stmp4.f * Ssh.f;
    Sch.f = Stmp4.f * Sch.f;

    Stmp1.f = Sfour_gamma_squared.f * Stmp1.f;
    Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

    Stmp2.ui = Ssine_pi_over_eight.ui & Stmp1.ui;
    Ssh.ui = ~Stmp1.ui & Ssh.ui;
    Ssh.ui = Ssh.ui | Stmp2.ui;
    Stmp2.ui = Scosine_pi_over_eight.ui & Stmp1.ui;
    Sch.ui = ~Stmp1.ui & Sch.ui;
    Sch.ui = Sch.ui | Stmp2.ui;

    Stmp1.f = Ssh.f * Ssh.f;
    Stmp2.f = Sch.f * Sch.f;
    Sc.f = Stmp2.f - Stmp1.f;
    Ss.f = Sch.f * Ssh.f;
    Ss.f = Ss.f + Ss.f;

    Stmp3.f = Stmp1.f + Stmp2.f;
    Ss11.f = Ss11.f * Stmp3.f;
    Ss21.f = Ss21.f * Stmp3.f;
    Ss31.f = Ss31.f * Stmp3.f;
    Ss11.f = Ss11.f * Stmp3.f;

    Stmp1.f = Ss.f * Ss21.f;
    Stmp2.f = Ss.f * Ss31.f;
    Ss21.f = Sc.f * Ss21.f;
    Ss31.f = Sc.f * Ss31.f;
    Ss21.f = Stmp2.f + Ss21.f;
    Ss31.f = Ss31.f - Stmp1.f;

    Stmp2.f = Ss.f * Ss.f;
    Stmp1.f = Ss33.f * Stmp2.f;
    Stmp3.f = Ss22.f * Stmp2.f;
    Stmp4.f = Sc.f * Sc.f;
    Ss22.f = Ss22.f * Stmp4.f;
    Ss33.f = Ss33.f * Stmp4.f;
    Ss22.f = Ss22.f + Stmp1.f;
    Ss33.f = Ss33.f + Stmp3.f;
    Stmp4.f = Stmp4.f - Stmp2.f;
    Stmp2.f = Ss32.f + Ss32.f;
    Ss32.f = Ss32.f * Stmp4.f;
    Stmp4.f = Sc.f * Ss.f;
    Stmp2.f = Stmp2.f * Stmp4.f;
    Stmp5.f = Stmp5.f * Stmp4.f;
    Ss22.f = Ss22.f + Stmp2.f;
    Ss32.f = Ss32.f - Stmp5.f;
    Ss33.f = Ss33.f - Stmp2.f;

    Stmp1.f = Ssh.f * Sqvvx.f;
    Stmp2.f = Ssh.f * Sqvvy.f;
    Stmp3.f = Ssh.f * Sqvvz.f;
    Ssh.f = Ssh.f * Sqvs.f;

    Sqvs.f = Sch.f * Sqvs.f;
    Sqvvx.f = Sch.f * Sqvvx.f;
    Sqvvy.f = Sch.f * Sqvvy.f;
    Sqvvz.f = Sch.f * Sqvvz.f;

    Sqvvx.f = Sqvvx.f + Ssh.f;
    Sqvs.f = Sqvs.f - Stmp1.f;
    Sqvvy.f = Sqvvy.f + Stmp3.f;
    Sqvvz.f = Sqvvz.f - Stmp2.f;
    Ssh.f = Ss31.f * Sone_half.f;
    Stmp5.f = Ss33.f - Ss11.f;

    Stmp2.f = Ssh.f * Ssh.f;
    Stmp1.ui = (Stmp2.f >= Stiny_number.f) ? 0xffffffff : 0;
    Ssh.ui = Stmp1.ui & Ssh.ui;
    Sch.ui = Stmp1.ui & Stmp5.ui;
    Stmp2.ui = ~Stmp1.ui & Sone.ui;
    Sch.ui = Sch.ui | Stmp2.ui;

    Stmp1.f = Ssh.f * Ssh.f;
    Stmp2.f = Sch.f * Sch.f;
    Stmp3.f = Stmp1.f + Stmp2.f;
    Stmp4.f = rsqrt(Stmp3.f);
    Ssh.f = Stmp4.f * Ssh.f;
    Sch.f = Stmp4.f * Sch.f;

    Stmp1.f = Sfour_gamma_squared.f * Stmp1.f;
    Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

    Stmp2.ui = Ssine_pi_over_eight.ui & Stmp1.ui;
    Ssh.ui = ~Stmp1.ui & Ssh.ui;
    Ssh.ui = Ssh.ui | Stmp2.ui;
    Stmp2.ui = Scosine_pi_over_eight.ui & Stmp1.ui;
    Sch.ui = ~Stmp1.ui & Sch.ui;
    Sch.ui = Sch.ui | Stmp2.ui;

    Stmp1.f = Ssh.f * Ssh.f;
    Stmp2.f = Sch.f * Sch.f;
    Sc.f = Stmp2.f - Stmp1.f;
    Ss.f = Sch.f * Ssh.f;
    Ss.f = Ss.f + Ss.f;

    Stmp3.f = Stmp1.f + Stmp2.f;
    Ss22.f = Ss22.f * Stmp3.f;
    Ss32.f = Ss32.f * Stmp3.f;
    Ss21.f = Ss21.f * Stmp3.f;
    Ss22.f = Ss22.f * Stmp3.f;

    Stmp1.f = Ss.f * Ss32.f;
    Stmp2.f = Ss.f * Ss21.f;
    Ss32.f = Sc.f * Ss32.f;
    Ss21.f = Sc.f * Ss21.f;
    Ss32.f = Stmp2.f + Ss32.f;
    Ss21.f = Ss21.f - Stmp1.f;

    Stmp2.f = Ss.f * Ss.f;
    Stmp1.f = Ss11.f * Stmp2.f;
    Stmp3.f = Ss33.f * Stmp2.f;
    Stmp4.f = Sc.f * Sc.f;
    Ss33.f = Ss33.f * Stmp4.f;
    Ss11.f = Ss11.f * Stmp4.f;
    Ss33.f = Ss33.f + Stmp1.f;
    Ss11.f = Ss11.f + Stmp3.f;
    Stmp4.f = Stmp4.f - Stmp2.f;
    Stmp2.f = Ss31.f + Ss31.f;
    Ss31.f = Ss31.f * Stmp4.f;
    Stmp4.f = Sc.f * Ss.f;
    Stmp2.f = Stmp2.f * Stmp4.f;
    Stmp5.f = Stmp5.f * Stmp4.f;
    Ss33.f = Ss33.f + Stmp2.f;
    Ss31.f = Ss31.f - Stmp5.f;
    Ss11.f = Ss11.f - Stmp2.f;

    Stmp1.f = Ssh.f * Sqvvx.f;
    Stmp2.f = Ssh.f * Sqvvy.f;
    Stmp3.f = Ssh.f * Sqvvz.f;
    Ssh.f = Ssh.f * Sqvs.f;

    Sqvs.f = Sch.f * Sqvs.f;
    Sqvvx.f = Sch.f * Sqvvx.f;
    Sqvvy.f = Sch.f * Sqvvy.f;
    Sqvvz.f = Sch.f * Sqvvz.f;

    Sqvvy.f = Sqvvy.f + Ssh.f;
    Sqvs.f = Sqvs.f - Stmp2.f;
    Sqvvz.f = Sqvvz.f + Stmp1.f;
    Sqvvx.f = Sqvvx.f - Stmp3.f;
  }

  Stmp2.f = Sqvs.f * Sqvs.f;
  Stmp1.f = Sqvvx.f * Sqvvx.f;
  Stmp2.f = Stmp1.f + Stmp2.f;
  Stmp1.f = Sqvvy.f * Sqvvy.f;
  Stmp2.f = Stmp1.f + Stmp2.f;
  Stmp1.f = Sqvvz.f * Sqvvz.f;
  Stmp2.f = Stmp1.f + Stmp2.f;

  Stmp1.f = rsqrt(Stmp2.f);
  Stmp4.f = Stmp1.f * Sone_half.f;
  Stmp3.f = Stmp1.f * Stmp4.f;
  Stmp3.f = Stmp1.f * Stmp3.f;
  Stmp3.f = Stmp2.f * Stmp3.f;
  Stmp1.f = Stmp1.f + Stmp4.f;
  Stmp1.f = Stmp1.f - Stmp3.f;

  Sqvs.f = Sqvs.f * Stmp1.f;
  Sqvvx.f = Sqvvx.f * Stmp1.f;
  Sqvvy.f = Sqvvy.f * Stmp1.f;
  Sqvvz.f = Sqvvz.f * Stmp1.f;

  Stmp1.f = Sqvvx.f * Sqvvx.f;
  Stmp2.f = Sqvvy.f * Sqvvy.f;
  Stmp3.f = Sqvvz.f * Sqvvz.f;
  Sv11.f = Sqvs.f * Sqvs.f;
  Sv22.f = Sv11.f - Stmp1.f;
  Sv33.f = Sv22.f - Stmp2.f;
  Sv33.f = Sv33.f + Stmp3.f;
  Sv22.f = Sv22.f + Stmp2.f;
  Sv22.f = Sv22.f - Stmp3.f;
  Sv11.f = Sv11.f + Stmp1.f;
  Sv11.f = Sv11.f - Stmp2.f;
  Sv11.f = Sv11.f - Stmp3.f;
  Stmp1.f = Sqvvx.f + Sqvvx.f;
  Stmp2.f = Sqvvy.f + Sqvvy.f;
  Stmp3.f = Sqvvz.f + Sqvvz.f;
  Sv32.f = Sqvs.f * Stmp1.f;
  Sv13.f = Sqvs.f * Stmp2.f;
  Sv21.f = Sqvs.f * Stmp3.f;
  Stmp1.f = Sqvvy.f * Stmp1.f;
  Stmp2.f = Sqvvz.f * Stmp2.f;
  Stmp3.f = Sqvvx.f * Stmp3.f;
  Sv12.f = Stmp1.f - Sv21.f;
  Sv23.f = Stmp2.f - Sv32.f;
  Sv31.f = Stmp3.f - Sv13.f;
  Sv21.f = Stmp1.f + Sv21.f;
  Sv32.f = Stmp2.f + Sv32.f;
  Sv13.f = Stmp3.f + Sv13.f;
  Stmp2.f = Sa12.f;
  Stmp3.f = Sa13.f;
  Sa12.f = Sv12.f * Sa11.f;
  Sa13.f = Sv13.f * Sa11.f;
  Sa11.f = Sv11.f * Sa11.f;
  Stmp1.f = Sv21.f * Stmp2.f;
  Sa11.f = Sa11.f + Stmp1.f;
  Stmp1.f = Sv31.f * Stmp3.f;
  Sa11.f = Sa11.f + Stmp1.f;
  Stmp1.f = Sv22.f * Stmp2.f;
  Sa12.f = Sa12.f + Stmp1.f;
  Stmp1.f = Sv32.f * Stmp3.f;
  Sa12.f = Sa12.f + Stmp1.f;
  Stmp1.f = Sv23.f * Stmp2.f;
  Sa13.f = Sa13.f + Stmp1.f;
  Stmp1.f = Sv33.f * Stmp3.f;
  Sa13.f = Sa13.f + Stmp1.f;

  Stmp2.f = Sa22.f;
  Stmp3.f = Sa23.f;
  Sa22.f = Sv12.f * Sa21.f;
  Sa23.f = Sv13.f * Sa21.f;
  Sa21.f = Sv11.f * Sa21.f;
  Stmp1.f = Sv21.f * Stmp2.f;
  Sa21.f = Sa21.f + Stmp1.f;
  Stmp1.f = Sv31.f * Stmp3.f;
  Sa21.f = Sa21.f + Stmp1.f;
  Stmp1.f = Sv22.f * Stmp2.f;
  Sa22.f = Sa22.f + Stmp1.f;
  Stmp1.f = Sv32.f * Stmp3.f;
  Sa22.f = Sa22.f + Stmp1.f;
  Stmp1.f = Sv23.f * Stmp2.f;
  Sa23.f = Sa23.f + Stmp1.f;
  Stmp1.f = Sv33.f * Stmp3.f;
  Sa23.f = Sa23.f + Stmp1.f;

  Stmp2.f = Sa32.f;
  Stmp3.f = Sa33.f;
  Sa32.f = Sv12.f * Sa31.f;
  Sa33.f = Sv13.f * Sa31.f;
  Sa31.f = Sv11.f * Sa31.f;
  Stmp1.f = Sv21.f * Stmp2.f;
  Sa31.f = Sa31.f + Stmp1.f;
  Stmp1.f = Sv31.f * Stmp3.f;
  Sa31.f = Sa31.f + Stmp1.f;
  Stmp1.f = Sv22.f * Stmp2.f;
  Sa32.f = Sa32.f + Stmp1.f;
  Stmp1.f = Sv32.f * Stmp3.f;
  Sa32.f = Sa32.f + Stmp1.f;
  Stmp1.f = Sv23.f * Stmp2.f;
  Sa33.f = Sa33.f + Stmp1.f;
  Stmp1.f = Sv33.f * Stmp3.f;
  Sa33.f = Sa33.f + Stmp1.f;

  Stmp1.f = Sa11.f * Sa11.f;
  Stmp4.f = Sa21.f * Sa21.f;
  Stmp1.f = Stmp1.f + Stmp4.f;
  Stmp4.f = Sa31.f * Sa31.f;
  Stmp1.f = Stmp1.f + Stmp4.f;

  Stmp2.f = Sa12.f * Sa12.f;
  Stmp4.f = Sa22.f * Sa22.f;
  Stmp2.f = Stmp2.f + Stmp4.f;
  Stmp4.f = Sa32.f * Sa32.f;
  Stmp2.f = Stmp2.f + Stmp4.f;

  Stmp3.f = Sa13.f * Sa13.f;
  Stmp4.f = Sa23.f * Sa23.f;
  Stmp3.f = Stmp3.f + Stmp4.f;
  Stmp4.f = Sa33.f * Sa33.f;
  Stmp3.f = Stmp3.f + Stmp4.f;

  Stmp4.ui = (Stmp1.f < Stmp2.f) ? 0xffffffff : 0;

  Stmp5.ui = Sa11.ui ^ Sa12.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sa11.ui = Sa11.ui ^ Stmp5.ui;
  Sa12.ui = Sa12.ui ^ Stmp5.ui;

  Stmp5.ui = Sa21.ui ^ Sa22.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sa21.ui = Sa21.ui ^ Stmp5.ui;
  Sa22.ui = Sa22.ui ^ Stmp5.ui;

  Stmp5.ui = Sa31.ui ^ Sa32.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sa31.ui = Sa31.ui ^ Stmp5.ui;
  Sa32.ui = Sa32.ui ^ Stmp5.ui;

  Stmp5.ui = Sv11.ui ^ Sv12.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sv11.ui = Sv11.ui ^ Stmp5.ui;
  Sv12.ui = Sv12.ui ^ Stmp5.ui;

  Stmp5.ui = Sv21.ui ^ Sv22.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sv21.ui = Sv21.ui ^ Stmp5.ui;
  Sv22.ui = Sv22.ui ^ Stmp5.ui;

  Stmp5.ui = Sv31.ui ^ Sv32.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sv31.ui = Sv31.ui ^ Stmp5.ui;
  Sv32.ui = Sv32.ui ^ Stmp5.ui;

  Stmp5.ui = Stmp1.ui ^ Stmp2.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Stmp1.ui = Stmp1.ui ^ Stmp5.ui;
  Stmp2.ui = Stmp2.ui ^ Stmp5.ui;

  Stmp5.f = -2.0f;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Stmp4.f = 1.0f;
  Stmp4.f = Stmp4.f + Stmp5.f;

  Sa12.f = Sa12.f * Stmp4.f;
  Sa22.f = Sa22.f * Stmp4.f;
  Sa32.f = Sa32.f * Stmp4.f;

  Sv12.f = Sv12.f * Stmp4.f;
  Sv22.f = Sv22.f * Stmp4.f;
  Sv32.f = Sv32.f * Stmp4.f;
  Stmp4.ui = (Stmp1.f < Stmp3.f) ? 0xffffffff : 0;

  Stmp5.ui = Sa11.ui ^ Sa13.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sa11.ui = Sa11.ui ^ Stmp5.ui;
  Sa13.ui = Sa13.ui ^ Stmp5.ui;

  Stmp5.ui = Sa21.ui ^ Sa23.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sa21.ui = Sa21.ui ^ Stmp5.ui;
  Sa23.ui = Sa23.ui ^ Stmp5.ui;

  Stmp5.ui = Sa31.ui ^ Sa33.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sa31.ui = Sa31.ui ^ Stmp5.ui;
  Sa33.ui = Sa33.ui ^ Stmp5.ui;

  Stmp5.ui = Sv11.ui ^ Sv13.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sv11.ui = Sv11.ui ^ Stmp5.ui;
  Sv13.ui = Sv13.ui ^ Stmp5.ui;

  Stmp5.ui = Sv21.ui ^ Sv23.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sv21.ui = Sv21.ui ^ Stmp5.ui;
  Sv23.ui = Sv23.ui ^ Stmp5.ui;

  Stmp5.ui = Sv31.ui ^ Sv33.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sv31.ui = Sv31.ui ^ Stmp5.ui;
  Sv33.ui = Sv33.ui ^ Stmp5.ui;

  Stmp5.ui = Stmp1.ui ^ Stmp3.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Stmp1.ui = Stmp1.ui ^ Stmp5.ui;
  Stmp3.ui = Stmp3.ui ^ Stmp5.ui;

  Stmp5.f = -2.0f;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Stmp4.f = 1.0f;
  Stmp4.f = Stmp4.f + Stmp5.f;

  Sa11.f = Sa11.f * Stmp4.f;
  Sa21.f = Sa21.f * Stmp4.f;
  Sa31.f = Sa31.f * Stmp4.f;

  Sv11.f = Sv11.f * Stmp4.f;
  Sv21.f = Sv21.f * Stmp4.f;
  Sv31.f = Sv31.f * Stmp4.f;
  Stmp4.ui = (Stmp2.f < Stmp3.f) ? 0xffffffff : 0;

  Stmp5.ui = Sa12.ui ^ Sa13.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sa12.ui = Sa12.ui ^ Stmp5.ui;
  Sa13.ui = Sa13.ui ^ Stmp5.ui;

  Stmp5.ui = Sa22.ui ^ Sa23.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sa22.ui = Sa22.ui ^ Stmp5.ui;
  Sa23.ui = Sa23.ui ^ Stmp5.ui;

  Stmp5.ui = Sa32.ui ^ Sa33.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sa32.ui = Sa32.ui ^ Stmp5.ui;
  Sa33.ui = Sa33.ui ^ Stmp5.ui;

  Stmp5.ui = Sv12.ui ^ Sv13.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sv12.ui = Sv12.ui ^ Stmp5.ui;
  Sv13.ui = Sv13.ui ^ Stmp5.ui;

  Stmp5.ui = Sv22.ui ^ Sv23.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sv22.ui = Sv22.ui ^ Stmp5.ui;
  Sv23.ui = Sv23.ui ^ Stmp5.ui;

  Stmp5.ui = Sv32.ui ^ Sv33.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Sv32.ui = Sv32.ui ^ Stmp5.ui;
  Sv33.ui = Sv33.ui ^ Stmp5.ui;

  Stmp5.ui = Stmp2.ui ^ Stmp3.ui;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Stmp2.ui = Stmp2.ui ^ Stmp5.ui;
  Stmp3.ui = Stmp3.ui ^ Stmp5.ui;

  Stmp5.f = -2.0f;
  Stmp5.ui = Stmp5.ui & Stmp4.ui;
  Stmp4.f = 1.0f;
  Stmp4.f = Stmp4.f + Stmp5.f;

  Sa13.f = Sa13.f * Stmp4.f;
  Sa23.f = Sa23.f * Stmp4.f;
  Sa33.f = Sa33.f * Stmp4.f;

  Sv13.f = Sv13.f * Stmp4.f;
  Sv23.f = Sv23.f * Stmp4.f;
  Sv33.f = Sv33.f * Stmp4.f;
  Su11.f = 1.0f;
  Su21.f = 0.0f;
  Su31.f = 0.0f;
  Su12.f = 0.0f;
  Su22.f = 1.0f;
  Su32.f = 0.0f;
  Su13.f = 0.0f;
  Su23.f = 0.0f;
  Su33.f = 1.0f;
  Ssh.f = Sa21.f * Sa21.f;
  Ssh.ui = (Ssh.f >= Ssmall_number.f) ? 0xffffffff : 0;

  Ssh.ui = Ssh.ui & Sa21.ui;

  Stmp5.f = 0.0f;
  Sch.f = Stmp5.f - Sa11.f;
  Sch.f = std::max(Sch.f, Sa11.f);
  Sch.f = std::max(Sch.f, Ssmall_number.f);
  Stmp5.ui = (Sa11.f >= Stmp5.f) ? 0xffffffff : 0;

  Stmp1.f = Sch.f * Sch.f;
  Stmp2.f = Ssh.f * Ssh.f;
  Stmp2.f = Stmp1.f + Stmp2.f;
  Stmp1.f = rsqrt(Stmp2.f);

  Stmp4.f = Stmp1.f * Sone_half.f;
  Stmp3.f = Stmp1.f * Stmp4.f;
  Stmp3.f = Stmp1.f * Stmp3.f;
  Stmp3.f = Stmp2.f * Stmp3.f;
  Stmp1.f = Stmp1.f + Stmp4.f;
  Stmp1.f = Stmp1.f - Stmp3.f;
  Stmp1.f = Stmp1.f * Stmp2.f;

  Sch.f = Sch.f + Stmp1.f;

  Stmp1.ui = ~Stmp5.ui & Ssh.ui;
  Stmp2.ui = ~Stmp5.ui & Sch.ui;
  Sch.ui = Stmp5.ui & Sch.ui;
  Ssh.ui = Stmp5.ui & Ssh.ui;
  Sch.ui = Sch.ui | Stmp1.ui;
  Ssh.ui = Ssh.ui | Stmp2.ui;

  Stmp1.f = Sch.f * Sch.f;
  Stmp2.f = Ssh.f * Ssh.f;
  Stmp2.f = Stmp1.f + Stmp2.f;
  Stmp1.f = rsqrt(Stmp2.f);

  Stmp4.f = Stmp1.f * Sone_half.f;
  Stmp3.f = Stmp1.f * Stmp4.f;
  Stmp3.f = Stmp1.f * Stmp3.f;
  Stmp3.f = Stmp2.f * Stmp3.f;
  Stmp1.f = Stmp1.f + Stmp4.f;
  Stmp1.f = Stmp1.f - Stmp3.f;

  Sch.f = Sch.f * Stmp1.f;
  Ssh.f = Ssh.f * Stmp1.f;

  Sc.f = Sch.f * Sch.f;
  Ss.f = Ssh.f * Ssh.f;
  Sc.f = Sc.f - Ss.f;
  Ss.f = Ssh.f * Sch.f;
  Ss.f = Ss.f + Ss.f;

  Stmp1.f = Ss.f * Sa11.f;
  Stmp2.f = Ss.f * Sa21.f;
  Sa11.f = Sc.f * Sa11.f;
  Sa21.f = Sc.f * Sa21.f;
  Sa11.f = Sa11.f + Stmp2.f;
  Sa21.f = Sa21.f - Stmp1.f;

  Stmp1.f = Ss.f * Sa12.f;
  Stmp2.f = Ss.f * Sa22.f;
  Sa12.f = Sc.f * Sa12.f;
  Sa22.f = Sc.f * Sa22.f;
  Sa12.f = Sa12.f + Stmp2.f;
  Sa22.f = Sa22.f - Stmp1.f;

  Stmp1.f = Ss.f * Sa13.f;
  Stmp2.f = Ss.f * Sa23.f;
  Sa13.f = Sc.f * Sa13.f;
  Sa23.f = Sc.f * Sa23.f;
  Sa13.f = Sa13.f + Stmp2.f;
  Sa23.f = Sa23.f - Stmp1.f;

  Stmp1.f = Ss.f * Su11.f;
  Stmp2.f = Ss.f * Su12.f;
  Su11.f = Sc.f * Su11.f;
  Su12.f = Sc.f * Su12.f;
  Su11.f = Su11.f + Stmp2.f;
  Su12.f = Su12.f - Stmp1.f;

  Stmp1.f = Ss.f * Su21.f;
  Stmp2.f = Ss.f * Su22.f;
  Su21.f = Sc.f * Su21.f;
  Su22.f = Sc.f * Su22.f;
  Su21.f = Su21.f + Stmp2.f;
  Su22.f = Su22.f - Stmp1.f;

  Stmp1.f = Ss.f * Su31.f;
  Stmp2.f = Ss.f * Su32.f;
  Su31.f = Sc.f * Su31.f;
  Su32.f = Sc.f * Su32.f;
  Su31.f = Su31.f + Stmp2.f;
  Su32.f = Su32.f - Stmp1.f;
  Ssh.f = Sa31.f * Sa31.f;
  Ssh.ui = (Ssh.f >= Ssmall_number.f) ? 0xffffffff : 0;

  Ssh.ui = Ssh.ui & Sa31.ui;

  Stmp5.f = 0.0f;
  Sch.f = Stmp5.f - Sa11.f;
  Sch.f = std::max(Sch.f, Sa11.f);
  Sch.f = std::max(Sch.f, Ssmall_number.f);
  Stmp5.ui = (Sa11.f >= Stmp5.f) ? 0xffffffff : 0;

  Stmp1.f = Sch.f * Sch.f;
  Stmp2.f = Ssh.f * Ssh.f;
  Stmp2.f = Stmp1.f + Stmp2.f;
  Stmp1.f = rsqrt(Stmp2.f);

  Stmp4.f = Stmp1.f * Sone_half.f;
  Stmp3.f = Stmp1.f * Stmp4.f;
  Stmp3.f = Stmp1.f * Stmp3.f;
  Stmp3.f = Stmp2.f * Stmp3.f;
  Stmp1.f = Stmp1.f + Stmp4.f;
  Stmp1.f = Stmp1.f - Stmp3.f;
  Stmp1.f = Stmp1.f * Stmp2.f;

  Sch.f = Sch.f + Stmp1.f;

  Stmp1.ui = ~Stmp5.ui & Ssh.ui;
  Stmp2.ui = ~Stmp5.ui & Sch.ui;
  Sch.ui = Stmp5.ui & Sch.ui;
  Ssh.ui = Stmp5.ui & Ssh.ui;
  Sch.ui = Sch.ui | Stmp1.ui;
  Ssh.ui = Ssh.ui | Stmp2.ui;

  Stmp1.f = Sch.f * Sch.f;
  Stmp2.f = Ssh.f * Ssh.f;
  Stmp2.f = Stmp1.f + Stmp2.f;
  Stmp1.f = rsqrt(Stmp2.f);

  Stmp4.f = Stmp1.f * Sone_half.f;
  Stmp3.f = Stmp1.f * Stmp4.f;
  Stmp3.f = Stmp1.f * Stmp3.f;
  Stmp3.f = Stmp2.f * Stmp3.f;
  Stmp1.f = Stmp1.f + Stmp4.f;
  Stmp1.f = Stmp1.f - Stmp3.f;

  Sch.f = Sch.f * Stmp1.f;
  Ssh.f = Ssh.f * Stmp1.f;

  Sc.f = Sch.f * Sch.f;
  Ss.f = Ssh.f * Ssh.f;
  Sc.f = Sc.f - Ss.f;
  Ss.f = Ssh.f * Sch.f;
  Ss.f = Ss.f + Ss.f;

  Stmp1.f = Ss.f * Sa11.f;
  Stmp2.f = Ss.f * Sa31.f;
  Sa11.f = Sc.f * Sa11.f;
  Sa31.f = Sc.f * Sa31.f;
  Sa11.f = Sa11.f + Stmp2.f;
  Sa31.f = Sa31.f - Stmp1.f;

  Stmp1.f = Ss.f * Sa12.f;
  Stmp2.f = Ss.f * Sa32.f;
  Sa12.f = Sc.f * Sa12.f;
  Sa32.f = Sc.f * Sa32.f;
  Sa12.f = Sa12.f + Stmp2.f;
  Sa32.f = Sa32.f - Stmp1.f;

  Stmp1.f = Ss.f * Sa13.f;
  Stmp2.f = Ss.f * Sa33.f;
  Sa13.f = Sc.f * Sa13.f;
  Sa33.f = Sc.f * Sa33.f;
  Sa13.f = Sa13.f + Stmp2.f;
  Sa33.f = Sa33.f - Stmp1.f;

  Stmp1.f = Ss.f * Su11.f;
  Stmp2.f = Ss.f * Su13.f;
  Su11.f = Sc.f * Su11.f;
  Su13.f = Sc.f * Su13.f;
  Su11.f = Su11.f + Stmp2.f;
  Su13.f = Su13.f - Stmp1.f;

  Stmp1.f = Ss.f * Su21.f;
  Stmp2.f = Ss.f * Su23.f;
  Su21.f = Sc.f * Su21.f;
  Su23.f = Sc.f * Su23.f;
  Su21.f = Su21.f + Stmp2.f;
  Su23.f = Su23.f - Stmp1.f;

  Stmp1.f = Ss.f * Su31.f;
  Stmp2.f = Ss.f * Su33.f;
  Su31.f = Sc.f * Su31.f;
  Su33.f = Sc.f * Su33.f;
  Su31.f = Su31.f + Stmp2.f;
  Su33.f = Su33.f - Stmp1.f;
  Ssh.f = Sa32.f * Sa32.f;
  Ssh.ui = (Ssh.f >= Ssmall_number.f) ? 0xffffffff : 0;

  Ssh.ui = Ssh.ui & Sa32.ui;

  Stmp5.f = 0.0f;
  Sch.f = Stmp5.f - Sa22.f;
  Sch.f = std::max(Sch.f, Sa22.f);
  Sch.f = std::max(Sch.f, Ssmall_number.f);
  Stmp5.ui = (Sa22.f >= Stmp5.f) ? 0xffffffff : 0;

  Stmp1.f = Sch.f * Sch.f;
  Stmp2.f = Ssh.f * Ssh.f;
  Stmp2.f = Stmp1.f + Stmp2.f;
  Stmp1.f = rsqrt(Stmp2.f);

  Stmp4.f = Stmp1.f * Sone_half.f;
  Stmp3.f = Stmp1.f * Stmp4.f;
  Stmp3.f = Stmp1.f * Stmp3.f;
  Stmp3.f = Stmp2.f * Stmp3.f;
  Stmp1.f = Stmp1.f + Stmp4.f;
  Stmp1.f = Stmp1.f - Stmp3.f;
  Stmp1.f = Stmp1.f * Stmp2.f;

  Sch.f = Sch.f + Stmp1.f;

  Stmp1.ui = ~Stmp5.ui & Ssh.ui;
  Stmp2.ui = ~Stmp5.ui & Sch.ui;
  Sch.ui = Stmp5.ui & Sch.ui;
  Ssh.ui = Stmp5.ui & Ssh.ui;
  Sch.ui = Sch.ui | Stmp1.ui;
  Ssh.ui = Ssh.ui | Stmp2.ui;

  Stmp1.f = Sch.f * Sch.f;
  Stmp2.f = Ssh.f * Ssh.f;
  Stmp2.f = Stmp1.f + Stmp2.f;
  Stmp1.f = rsqrt(Stmp2.f);

  Stmp4.f = Stmp1.f * Sone_half.f;
  Stmp3.f = Stmp1.f * Stmp4.f;
  Stmp3.f = Stmp1.f * Stmp3.f;
  Stmp3.f = Stmp2.f * Stmp3.f;
  Stmp1.f = Stmp1.f + Stmp4.f;
  Stmp1.f = Stmp1.f - Stmp3.f;

  Sch.f = Sch.f * Stmp1.f;
  Ssh.f = Ssh.f * Stmp1.f;

  Sc.f = Sch.f * Sch.f;
  Ss.f = Ssh.f * Ssh.f;
  Sc.f = Sc.f - Ss.f;
  Ss.f = Ssh.f * Sch.f;
  Ss.f = Ss.f + Ss.f;

  Stmp1.f = Ss.f * Sa21.f;
  Stmp2.f = Ss.f * Sa31.f;
  Sa21.f = Sc.f * Sa21.f;
  Sa31.f = Sc.f * Sa31.f;
  Sa21.f = Sa21.f + Stmp2.f;
  Sa31.f = Sa31.f - Stmp1.f;

  Stmp1.f = Ss.f * Sa22.f;
  Stmp2.f = Ss.f * Sa32.f;
  Sa22.f = Sc.f * Sa22.f;
  Sa32.f = Sc.f * Sa32.f;
  Sa22.f = Sa22.f + Stmp2.f;
  Sa32.f = Sa32.f - Stmp1.f;

  Stmp1.f = Ss.f * Sa23.f;
  Stmp2.f = Ss.f * Sa33.f;
  Sa23.f = Sc.f * Sa23.f;
  Sa33.f = Sc.f * Sa33.f;
  Sa23.f = Sa23.f + Stmp2.f;
  Sa33.f = Sa33.f - Stmp1.f;

  Stmp1.f = Ss.f * Su12.f;
  Stmp2.f = Ss.f * Su13.f;
  Su12.f = Sc.f * Su12.f;
  Su13.f = Sc.f * Su13.f;
  Su12.f = Su12.f + Stmp2.f;
  Su13.f = Su13.f - Stmp1.f;

  Stmp1.f = Ss.f * Su22.f;
  Stmp2.f = Ss.f * Su23.f;
  Su22.f = Sc.f * Su22.f;
  Su23.f = Sc.f * Su23.f;
  Su22.f = Su22.f + Stmp2.f;
  Su23.f = Su23.f - Stmp1.f;

  Stmp1.f = Ss.f * Su32.f;
  Stmp2.f = Ss.f * Su33.f;
  Su32.f = Sc.f * Su32.f;
  Su33.f = Sc.f * Su33.f;
  Su32.f = Su32.f + Stmp2.f;
  Su33.f = Su33.f - Stmp1.f;
  // end

  u11 = Su11.f;
  u21 = Su21.f;
  u31 = Su31.f;
  u12 = Su12.f;
  u22 = Su22.f;
  u32 = Su32.f;
  u13 = Su13.f;
  u23 = Su23.f;
  u33 = Su33.f;

  v11 = Sv11.f;
  v21 = Sv21.f;
  v31 = Sv31.f;
  v12 = Sv12.f;
  v22 = Sv22.f;
  v32 = Sv32.f;
  v13 = Sv13.f;
  v23 = Sv23.f;
  v33 = Sv33.f;

  sigma1 = Sa11.f;
  sigma2 = Sa22.f;
  sigma3 = Sa33.f;
  // output
}

}  // namespace SifakisSVD
