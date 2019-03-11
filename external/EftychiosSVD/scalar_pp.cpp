#include <cmath>
#include <immintrin.h>
#include <algorithm>

float rsqrt(const float f) {
  float buf[4];
  buf[0] = f;
  __m128 v = _mm_loadu_ps(buf);
  v = _mm_rsqrt_ss(v);
  _mm_storeu_ps(buf, v);
  return buf[0];
}

inline void svd(
    float *a11, float *a12, float *a13,
    float *a21, float *a22, float *a23,
    float *a31, float *a32, float *a33,
    float *u11, float *u12, float *u13,
    float *u21, float *u22, float *u23,
    float *u31, float *u32, float *u33,
    float *v11, float *v12, float *v13,
    float *v21, float *v22, float *v23,
    float *v31, float *v32, float *v33,
    float *sigma1, float *sigma2, float *sigma3
    ) {
  const float Four_Gamma_Squared = sqrt(8.) + 3.;
  const float Sine_Pi_Over_Eight = .5 * sqrt(2. - sqrt(2.));
  const float Cosine_Pi_Over_Eight = .5 * sqrt(2. + sqrt(2.));
  const unsigned One_Mask = 0xffffffff;

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

  Sfour_gamma_squared.f = Four_Gamma_Squared;
  Ssine_pi_over_eight.f = Sine_Pi_Over_Eight;
  Scosine_pi_over_eight.f = Cosine_Pi_Over_Eight;
  Sone_half.f = .5;
  Sone.f = 1.;
  Stiny_number.f = 1.e-20;
  Ssmall_number.f = 1.e-12;

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

  for (int index = 0; index < 1; index += 1) {
    Sa11.f = a11[index];
    Sa21.f = a21[index];
    Sa31.f = a31[index];
    Sa12.f = a12[index];
    Sa22.f = a22[index];
    Sa32.f = a32[index];
    Sa13.f = a13[index];
    Sa23.f = a23[index];
    Sa33.f = a33[index];

    {

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

      {

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

        Sqvs.f = 1.;
        Sqvvx.f = 0.;
        Sqvvy.f = 0.;
        Sqvvz.f = 0.;

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

        for (int sweep = 1; sweep <= 4; sweep++) {
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
      {
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
      }
    }

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

    Stmp5.f = -2.;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp4.f = 1.;
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

    Stmp5.f = -2.;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp4.f = 1.;
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

    Stmp5.f = -2.;
    Stmp5.ui = Stmp5.ui & Stmp4.ui;
    Stmp4.f = 1.;
    Stmp4.f = Stmp4.f + Stmp5.f;

    Sa13.f = Sa13.f * Stmp4.f;
    Sa23.f = Sa23.f * Stmp4.f;
    Sa33.f = Sa33.f * Stmp4.f;

    Sv13.f = Sv13.f * Stmp4.f;
    Sv23.f = Sv23.f * Stmp4.f;
    Sv33.f = Sv33.f * Stmp4.f;
    Su11.f = 1.;
    Su21.f = 0.;
    Su31.f = 0.;
    Su12.f = 0.;
    Su22.f = 1.;
    Su32.f = 0.;
    Su13.f = 0.;
    Su23.f = 0.;
    Su33.f = 1.;
    Ssh.f = Sa21.f * Sa21.f;
    Ssh.ui = (Ssh.f >= Ssmall_number.f) ? 0xffffffff : 0;

    Ssh.ui = Ssh.ui & Sa21.ui;

    Stmp5.f = 0.;
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

    Stmp5.f = 0.;
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

    Stmp5.f = 0.;
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

    u11[index] = Su11.f;
    u21[index] = Su21.f;
    u31[index] = Su31.f;
    u12[index] = Su12.f;
    u22[index] = Su22.f;
    u32[index] = Su32.f;
    u13[index] = Su13.f;
    u23[index] = Su23.f;
    u33[index] = Su33.f;

    v11[index] = Sv11.f;
    v21[index] = Sv21.f;
    v31[index] = Sv31.f;
    v12[index] = Sv12.f;
    v22[index] = Sv22.f;
    v32[index] = Sv32.f;
    v13[index] = Sv13.f;
    v23[index] = Sv23.f;
    v33[index] = Sv33.f;

    sigma1[index] = Sa11.f;
    sigma2[index] = Sa22.f;
    sigma3[index] = Sa33.f;
  }
}

int main() {
}
