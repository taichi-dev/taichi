#pragma once

#include <taichi/math/linalg.h>

TC_NAMESPACE_BEGIN

const real stefan_boltzmann_constant = 5.670373e-8f;
const double boltzmann_constant = 1.38064852e-23;
const real speed_of_light = 3e8f;
const double plank_constant = 6.63e-34;

inline real luminance(Vector3 v) {
    float std_y_weight[3] = { 0.212671f, 0.715160f, 0.072169f };
    float lum = 0;
    for (int i = 0; i < 3; i++) {
        lum += std_y_weight[i] * v[i];
    }
    return lum; // * 683.0f
}

TC_NAMESPACE_END

