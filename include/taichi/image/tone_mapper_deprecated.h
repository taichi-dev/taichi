/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

/*

#include <taichi/visualization/image_buffer.h>
#include <taichi/math/math.h>
#include <taichi/physics/spectrum.h>
#include <taichi/physics/physics_constants.h>

TC_NAMESPACE_BEGIN

class ToneMapper {
public:
    static Array2D<Vector3> apply(const Array2D<Vector3> &input) {
        int width = input.get_width();
        int height = input.get_height();
        Array2D<Vector3> output(width, height);
        real sum = 0.0f;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < 3; k++) {
                    sum += input[i][j][k];
                }
            }
        }
        real ave = sum / (width * height * 3) * 5.0f;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < 3; k++) {
                    real inp = input[i][j][k] / ave;
                    output[i][j][k] = pow(inp / (inp + 1), 1);
                }
            }
        }
        return output;
    }
};

class AverageToOne {
public:
    static Array2D<Vector3> apply(const Array2D<Vector3> &input) {
        int width = input.get_width();
        int height = input.get_height();
        Array2D<Vector3> output(width, height);
        real sum_lum = 0.0f;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                sum_lum += luminance(input[i][j]);
            }
        }
        real ave = sum_lum / (width * height) / 0.18f;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                output[i][j] = input[i][j] / ave;
            }
        }
        return output;
    }
};

class GammaCorrection {
public:
    static Array2D<Vector3> apply(const Array2D<Vector3> &input) {
        int width = input.get_width();
        int height = input.get_height();
        Array2D<Vector3> output(width, height);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < 3; k++) {
                    output[i][j][k] = pow(input[i][j][k], 1);
                }
            }
        }
        return output;
    }
};

class ColorPreservingToneMapper {
public:
    static Array2D<Vector3> apply(const Array2D<Vector3> &input) {
        int width = input.get_width();
        int height = input.get_height();
        Array2D<Vector3> output(width, height);
        real sum = 0.0f;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < 3; k++) {
                    sum += input[i][j][k];
                }
            }
        }
        real ave = sum / (width * height * 3) * 5.0f;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                float lum = 0;
                for (int k = 0; k < 3; k++) {
                    lum += input[i][j][k] / ave;
                }
                float scale;
                scale = log(lum) / (lum + 1);
                for (int k = 0; k < 3; k++) {
                    output[i][j][k] = scale * input[i][j][k] / ave;
                    output[i][j][k] = pow(output[i][j][k], 1);
                }
            }
        }
        return output;
    }
};

class HeatToneMapper {
public:
    static Array2D<Vector3> apply(const Array2D<Vector3> &input) {
        int width = input.get_width();
        int height = input.get_height();
        Array2D<Vector3> output(width, height);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                real temp = max(input[i][j][0], eps);
                real x = temp / 10;
                x = (log(x + 0.1f) - log(0.1f)) / 5;
                x = x / (x + 1);
                output[i][j] = Vector3(x, 0, 1 - x);
            }
        }
        return output;
    }
};

class PhysicallyBasedHeatToneMapper {
public:
    static Array2D<Vector3> apply(const Array2D<Vector3> &input) {
        int width = input.get_width();
        int height = input.get_height();
        Array2D<Vector3> output(width, height);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                real temp = max(input[i][j][0], eps);
                output[i][j] = Spectrum::get_instance().sample(temp);
            }
        }
        return output;
    }
};

class LogLuminance {
public:
    static Array2D<Vector3> apply(const Array2D<Vector3> &input) {
        int width = input.get_width();
        int height = input.get_height();
        Array2D<Vector3> output(width, height);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                float lum = max(luminance(input[i][j]) * 1e5f, 1.0_f);
                float scale = log(lum) / lum;
                output[i][j] = input[i][j] * scale;
            }
        }
        return output;
    }
};

class MaxToWhite {
public:
    static Array2D<Vector3> apply(const Array2D<Vector3> &input) {
        int width = input.get_width();
        int height = input.get_height();
        Array2D<Vector3> output(width, height);
        real max_comp = 1e-7f;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                max_comp = max(max_comp, max_component(input[i][j]));
            }
        }
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                output[i][j] = input[i][j] / max_comp;
            }
        }
        return output;
    }
};

class PBRTToneMapper {
public:
    static Array2D<Vector3> apply(Array2D<Vector3> input) {
        int width = input.get_width();
        int height = input.get_height();
        Array2D<Vector3> output(width, height);
        real max_display_y = 100;
        real max_y = -1e10f;
        Vector3 average = input.get_average();
        float exposure_control = 683 * 0.18f / luminance(average);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                input[i][j] *= exposure_control;
                max_y = max(max_y, luminance(input[i][j]));
            }
        }
        float inv_y2;
        if (max_y <= 0.0f) {
            float ywa = 0.0f;
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    real y = luminance(input[i][j]);
                    if (y > 0) {
                        ywa += logf(y);
                    }
                }
            }
            ywa = expf(ywa / (width * height));
            ywa /= 683;
            inv_y2 = 1.f / (ywa * ywa);
        }
        else {
            inv_y2 = 1.0_f / (max_y * max_y);
        }
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                float ys = luminance(input[i][j]) / 683;
                float s = max_display_y / 683.0f * (1 + ys * inv_y2) / (1.0_f +
ys);
                // float s = (log(ys) + 5) / (log(max_y) + 5) / ys;
                // if (!(s >= 0)) s = 0;
                Vector3 c = input[i][j] * s * 683.0f / max_display_y;
                if (max_component(c) > 1.0_f) {
                    c /= max_component(c);
                }
                output[i][j] = pow(c, 1 / 2.2f);
            }
        }
        return output;
    }
};


TC_NAMESPACE_END

*/
