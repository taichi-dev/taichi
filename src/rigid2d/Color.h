/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include "Constants.h"

inline float randf();

struct HSB3f {
  float h, s, b;
  HSB3f(float h, float s, float b) : h(h), s(s), b(b) {}
};

struct RGB3f {
  float r, g, b;
  RGB3f() { r = g = b = 1.0_f; }
  RGB3f(float r, float g, float b) : r(r), g(g), b(b) {}
  RGB3f(HSB3f hsb) {
    float H = hsb.h / 60.0_f, S = hsb.s, B = hsb.b;
    if (S == 0.0_f) {
      *this = RGB3f(B, B, B);
      return;
    }

    float Hi = floor(H);
    float f = H - Hi;
    float p = B * (1 - S);
    float q = B * (1 - f * S);
    float t = B * (1 - (1 - f) * S);
    float v = B;

    switch ((int)Hi) {
      case 0:
        *this = RGB3f(v, t, p);
        break;
      case 1:
        *this = RGB3f(q, v, p);
        break;
      case 2:
        *this = RGB3f(p, v, t);
        break;
      case 3:
        *this = RGB3f(p, q, v);
        break;
      case 4:
        *this = RGB3f(t, p, v);
        break;
      default:
        *this = RGB3f(v, p, q);
    }
  }
  static RGB3f RandomBrightColor() {
    return HSB3f(randf() * 360.0_f, 0.8f, 0.8f);
  }
  operator float *() { return (float *)this; }
};

class Colors {
 public:
  static RGB3f White;
  static RGB3f Black;
  static RGB3f RandomBrightColor() {
    return HSB3f(randf() * 360.0_f, 0.8f, 0.8f);
  }
};
