#pragma once

#include <vector>
#include "math_utils.h"
#include "common/utils.h"

TC_NAMESPACE_BEGIN

    using namespace std;
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

    inline Vector3 cross(const Vector3 &a, const Vector3 &b) {
        return glm::cross(a, b);
    }

    inline real dot(const Vector3 &a, const Vector3 &b) {
        return glm::dot(a, b);
    }

    inline real length(const Vector3 &a, const Vector3 &b) {
        return sqrt(dot(a, b));
    }

    inline Vector3 normalize(const Vector3 &a) {
        return glm::normalize(a);
    }

    inline Vector3 normalized(const Vector3 &a) {
        return normalize(a);
    }

    inline Vector3 abs(const Vector3 &a) {
        return Vector3(std::abs(a.x), std::abs(a.y), std::abs(a.z));
    }

    inline Vector3 set_up(const Vector3 &a, const Vector3 &y) {
        Vector3 x, z;
        if (std::abs(y.y) > 1.0f - eps) {
            x = Vector3(1, 0, 0);
        } else {
            x = normalize(cross(y, Vector3(0, 1, 0)));
        }
        z = cross(x, y);
        return a.x * x + a.y * y + a.z * z;
    }

    inline Vector3 multiply_matrix4(Matrix4 m, Vector3 v, real w) {
        Vector4 tmp(v, w);
        tmp = m * tmp;
        return Vector3(tmp.x, tmp.y, tmp.z);
    }

/*
inline Vector3 random_diffuse(const Vector3 &normal) {
	real xzs = rand();
	real xz = sqrt(xzs), y = sqrt(1 - xzs);
	real phi = rand() * pi * 2;
	return set_up(Vector3(xz * cos(phi), y, xz * sin(phi)), normal);
}
*/

    inline Vector3 random_diffuse(const Vector3 &normal, real u, real v) {
        if (u > v) {
            swap(u, v);
        }
        if (v < eps) {
            v = eps;
        }
        u /= v;
        real xz = v, y = sqrt(1 - v * v);
        real phi = u * pi * 2;
        return set_up(Vector3(xz * cos(phi), y, xz * sin(phi)), normal);
    }

    inline Vector3 reflect(const Vector3 &d, const Vector3 &n) {
        return glm::reflect(d, n);
    }

    inline Vector3 random_diffuse(const Vector3 &normal) {
        return random_diffuse(normal, rand(), rand());
    }

    inline Vector4 pow(const Vector4 &v, const float &p) {
        return Vector4(
                ::pow(v[0], p),
                ::pow(v[1], p),
                ::pow(v[2], p),
                ::pow(v[3], p)
        );
    }

    inline Vector3 pow(const Vector3 &v, const float &p) {
        return Vector3(
                ::pow(v[0], p),
                ::pow(v[1], p),
                ::pow(v[2], p)
        );
    }


    inline real max_component(const Vector3 &v) {
        return max(v.x, max(v.y, v.z));
    }

    inline double max_component(const Vector3d &v) {
        return max(v.x, max(v.y, v.z));
    }

#define CV_ON
#ifdef CV_ON
#define CV(v) if (!is_normal(v)) {for (int i = 0; i < 1; i++) printf("Abnormal value %s (Ln %d)\n", #v, __LINE__); taichi::print(v); puts("");}
#else
#define CV(v) 
#endif

#define abnormal(v) (!is_normal(v))

    inline bool is_normal(float m) {
        return -1e30f < m && m < 1e30f;
    }

    inline bool is_normal(vec2 v) {
        return is_normal(v[0]) && is_normal(v[1]);
    }

    inline bool is_normal(Vector3 v) {
        return is_normal(v[0]) && is_normal(v[1]) && is_normal(v[2]);
    }

    inline bool is_normal(mat2 m) {
        return is_normal(m[0][0]) && is_normal(m[0][1]) &&
               is_normal(m[1][0]) && is_normal(m[1][1]);
    }

    inline Vector2 clamp(const Vector2 &v) {
        return Vector2(clamp(v[0]), clamp(v[1]));
    }

    inline float cross(const Vector2 &a, const Vector2 &b) {
        return a.x * b.y - a.y * b.x;
    }

    inline float matrix_norm_squared(const Matrix2 &a) {
        float sum = 0.0f;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                sum += a[i][j] * a[i][j];
            }
        }
        return sum;
    }

    inline float matrix_norm_squared(const Matrix3 &a) {
        float sum = 0.0f;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                sum += a[i][j] * a[i][j];
            }
        }
        return sum;
    }

    inline float frobenius_norm(const Matrix2 &a) {
        return sqrt(a[0][0] * a[0][0] + a[0][1] * a[0][1] + a[1][0] * a[1][0] + a[1][1] * a[1][1]);
    }

    inline float frobenius_norm(const Matrix3 &a) {
        return sqrt(a[0][0] * a[0][0] + a[0][1] * a[0][1] + a[1][0] * a[1][0] + a[1][1] * a[1][1]);
    }


    inline bool intersect(const Vector2 &a, const Vector2 &b, const Vector2 &c, const Vector2 &d) {
        if (cross(c - a, b - a) * cross(b - a, d - a) > 0 && cross(a - d, c - d) * cross(c - d, b - d) > 0) {
            return true;
        } else {
            return false;
        }
    }

    inline float nearest_distance(const Vector2 &p, const Vector2 &a, const Vector2 &b) {
        float ab = length(a - b);
        Vector2 dir = (b - a) / ab;
        float pos = clamp(dot(p - a, dir), 0.0f, ab);
        return length(a + pos * dir - p);
    }

    inline float nearest_distance(const Vector2 &p, const std::vector<Vector2> polygon) {
        float dist = std::numeric_limits<float>::infinity();
        for (int i = 0; i < (int) polygon.size(); i++) {
            dist = std::min(dist, nearest_distance(p, polygon[i], polygon[(i + 1) % polygon.size()]));
        }
        return dist;
    }

    inline bool inside_polygon(const Vector2 &p, const std::vector<Vector2> polygon) {
        int count = 0;
        static const Vector2 q(123532, 532421123);
        for (int i = 0; i < (int) polygon.size(); i++) {
            count += intersect(p, q, polygon[i], polygon[(i + 1) % polygon.size()]);
        }
        return count % 2 == 1;
    }

    inline std::vector<Vector2>
    points_inside_polygon(std::vector<float> x_range, std::vector<float> y_range, const std::vector<Vector2> polygon) {
        std::vector<Vector2> ret;
        for (float x = x_range[0]; x < x_range[1]; x += x_range[2]) {
            for (float y = y_range[0]; y < y_range[1]; y += y_range[2]) {
                Vector2 p(x, y);
                if (inside_polygon(p, polygon)) {
                    ret.push_back(p);
                }
            }
        }
        return ret;
    }

    inline std::vector<Vector2>
    points_inside_sphere(std::vector<float> x_range, std::vector<float> y_range, const Vector2 &center, float radius) {
        std::vector<Vector2> ret;
        for (float x = x_range[0]; x < x_range[1]; x += x_range[2]) {
            for (float y = y_range[0]; y < y_range[1]; y += y_range[2]) {
                Vector2 p(x, y);
                if (length(p - center) < radius) {
                    ret.push_back(p);
                }
            }
        }
        return ret;
    }

TC_NAMESPACE_END
