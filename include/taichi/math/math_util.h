#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat2x2.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/vector_relational.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include <taichi/common/util.h>

TC_NAMESPACE_BEGIN

using glm::vec2;
using glm::ivec2;
using glm::vec3;
using glm::vec4;
using glm::mat2;
using glm::mat3;
using glm::mat4;

typedef glm::vec2 Vector2;
typedef glm::vec3 Vector3;
typedef glm::vec4 Vector4;

typedef glm::ivec2 Vector2i;
typedef glm::ivec3 Vector3i;
typedef glm::ivec4 Vector4i;

typedef glm::vec2 Vector2f;
typedef glm::vec3 Vector3f;
typedef glm::vec4 Vector4f;

typedef glm::dvec2 Vector2d;
typedef glm::dvec3 Vector3d;
typedef glm::dvec4 Vector4d;

typedef float real;
typedef glm::mat2 Matrix2;
typedef glm::mat3 Matrix3;
typedef glm::mat4 Matrix4;

const real pi{ acosf(-1.0f) };

const real eps = 1e-6f;

#undef max
#undef min

//template<typename T>
//inline T max(const T &a, const T &b) {
//    return a < b ? a : b;
//}
//
//template<typename T>
//inline T min(const T &a, const T &b) {
//    return a < b ? a : b;
//}

using glm::max;

template<typename T>
inline T abs(const T &a) {
    return std::abs(a);
}

template<typename T>
inline T clamp(T a, T min, T max) {
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

template<typename T>
inline T clamp(T a) {
    if (a < 0) return 0;
    if (a > 1) return 1;
    return a;
}

template<typename T, typename V>
inline V lerp(T a, V x_0, V x_1) {
    return (T(1) - a) * x_0 + a * x_1;
}

inline bool inside_unit_cube(const Vector3 &p) {
    return 0 <= p.x && p.x < 1 && 0 <= p.y && p.y < 1 && 0 <= p.z && p.z < 1;
}

template<typename T>
T sqr(const T &a) {
    return a * a;
}

template<typename T>
T cube(const T &a) {
    return a * a * a;
}

inline int sgn(float a) {
    if (a < -eps)
        return -1;
    else if (a > eps)
        return 1;
    return 0;
}

inline int sgn(double a) {
    if (a < -eps)
        return -1;
    else if (a > eps)
        return 1;
    return 0;
}

// inline float frand() { return (float)rand() / (RAND_MAX + 1); }
inline float rand() {
    static unsigned int x = 123456789, y = 362436069, z = 521288629, w = 88675123;
    unsigned int t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8))) * (1.0f / 4294967296.0f);
}

inline vec3 sample_sphere(float u, float v) {
    float x = u * 2 - 1;
    float phi = v * 2 * pi;
    float yz = sqrt(1 - x * x);
    return vec3(x, yz * cos(phi), yz * sin(phi));
}


inline float catmull_rom(float f_m_1, float f_0, float f_1, float f_2,
    float x_r) {
    float s = (f_1 - f_0);
    float s_0 = (f_1 - f_m_1) / 2.0f;
    float s_1 = (f_2 - f_0) / 2.0f;
    s_0 = s_0 * (s_1 * s > 0);
    s_1 = s_1 * (s_1 * s > 0);
    return f_0 + x_r * s_0 + (-3 * f_0 + 3 * f_1 - 2 * s_0 - s_1) * x_r * x_r +
        (2 * f_0 - 2 * f_1 + s_0 + s_1) * x_r * x_r * x_r;
}

inline float catmull_rom(float *pf_m_1, float x_r) {
    return catmull_rom(*pf_m_1, *(pf_m_1 + 1), *(pf_m_1 + 2), *(pf_m_1 + 3), x_r);
}

inline void print(std::string v) {
    printf("%s\n", v.c_str());
}

inline void print(float v) {
    printf("%f\n", v);
}

inline void print(int v) {
    printf("%d\n", v);
}

inline void print(unsigned int v) {
    printf("%u\n", v);
}

#ifndef WIN32
inline void print(size_t v) {
    printf("%lld\n", (long long)v);
}
#endif

inline void print(long long v) {
    std::cout << v << std::endl;
}

inline void print(unsigned long long v) {
    std::cout << v << std::endl;
}


inline void print(double v) {
    printf("%f\n", v);
}

inline void print(const mat2 &v) {
    printf("\n%f %f\n%f %f\n", v[0][0], v[1][0], v[0][1], v[1][1]);
}

inline void print(const Vector2 &v) {
    printf("%f %f\n", v[0], v[1]);
}

inline void print(const Vector3 &v) {
    printf("%f %f %f\n", v[0], v[1], v[2]);
}

inline void print(const Vector4 &v) {
    printf("%f %f %f %f\n", v[0], v[1], v[2], v[3]);
}

inline void print(const Vector2d &v) {
    printf("%f %f\n", v[0], v[1]);
}

inline void print(const Vector3d &v) {
    printf("%f %f %f\n", v[0], v[1], v[2]);
}

inline void print(const Vector4d &v) {
    printf("%f %f %f %f\n", v[0], v[1], v[2], v[3]);
}

inline void print(const Vector2i &v) {
    printf("%d %d\n", v[0], v[1]);
}

inline void print(const Vector3i &v) {
    printf("%d %d %d\n", v[0], v[1], v[2]);
}

inline void print(const Vector4i &v) {
    printf("%d %d %d %d\n", v[0], v[1], v[2], v[3]);
}

inline void print(const mat4 &v) {
    printf("\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", v[j][i]);
        }
        printf("\n");
    }
}

inline void print(const mat3 &v) {
    printf("\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", v[j][i]);
        }
        printf("\n");
    }
}

inline int is_prime(int a) {
    assert(a >= 2);
    for (int i = 2; i * i <= a; i++) {
        if (a % i == 0) return false;
    }
    return true;
}

template<typename T>
inline T hypot2(const T &x, const T &y) {
    return x * x + y * y;
}

inline float pow(const float &a, const float &b) {
    return ::pow(a, b);
}

inline double pow(const double &a, const double &b) {
    return ::pow(a, b);
}

//#define rand frand


TC_NAMESPACE_END

