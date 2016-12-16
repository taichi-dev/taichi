#pragma once

#include <taichi/common/util.h>
#include <taichi/math/linalg.h>
#include <vector>

TC_NAMESPACE_BEGIN

class Ray {
public:
	Ray() {};
	Ray(Vector3 orig, Vector3 dir, real time=0) : orig(orig),
		dir(dir), dist(DIST_INFINITE), time(time) {
		triangle_id = -1;
	}

	Vector3 orig, dir;
	real time, dist;
	int triangle_id;
	Vector3 geometry_normal;
	real u, v;

	const static float DIST_INFINITE;
};

struct Triangle {
	Triangle() {};
	Vector3 v[3];
	Vector3 v10, v20, normal;
	Vector3 n0, n1, n2;
    Vector2 uv0, uv1, uv2;
	Vector3 n10, n20;
	int id;
	real area;
	real temperature;
	real heat_capacity;
	real specific_heat;
	bool reflective;
	bool refractive;
	Triangle(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2, int id) {
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
		v10 = v1 - v0;
		v20 = v2 - v0;
		normal = normalized(cross(v10, v20));
		n10 = normal;
		n20 = normal;
		this->id = id;
		area = 0.5f * length(cross(v[1] - v[0], v[2] - v[0]));
	}
	Triangle(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2,
			 const Vector3 &n0, const Vector3 &n1, const Vector3 &n2,
			 const Vector2 &uv0, const Vector2 &uv1, const Vector2 &uv2, int id) {
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
		this->n0 = n0;
		this->n1 = n1;
		this->n2 = n2;
		this->uv0 = uv0;
		this->uv1 = uv1;
		this->uv2 = uv2;
		v10 = v1 - v0;
		v20 = v2 - v0;
		n10 = n1 - n0;
		n20 = n2 - n0;
		this->id = id;
		normal = normalized(cross(v10, v20));
		area = 0.5f * length(cross(v[1] - v[0], v[2] - v[0]));
	}
	void get_coord(const Ray &ray, real dist, real &coord_u, real &coord_v) const {
		const Vector3 inter_local = ray.orig + ray.dir * dist - v[0];
		const Vector3 u = v10, v = v20;
		real uv = dot(u, v), vv = dot(v, v), wu = dot(inter_local, u), uu = dot(u, u), wv = dot(inter_local, v);
		real dom = uv * uv - uu * vv;
		coord_u = (uv * wv - vv * wu) / dom;
		coord_v = (uv * wu - uu * wv) / dom;
	}
	void intersect(Ray &ray) {
		const Vector3 &orig = ray.orig;
		const Vector3 &dir = ray.dir;
		float dir_n = dot(dir, normal);
		float dist_n = dot(v[0] - orig, normal);
		float dist = dist_n / dir_n;
		if (dist <= 0.0f) {
			return;
		}
		else {
			if (dist > 0 && dist < ray.dist) {
				real coord_u, coord_v;
				get_coord(ray, dist, coord_u, coord_v);
				if (coord_u >= 0 && coord_v >= 0 && coord_u + coord_v <= 1) {
					ray.dist = dist;
					ray.triangle_id = id;
					ray.u = coord_u;
					ray.v = coord_v;
				}
			}
		}
	}
	Vector3 get_normal(real u, real v) const {
		return normalized(n0 + u * n10 + v * n20);
	}
	Vector2 get_uv(real u, real v) const {
		return (1.0f - u - v) * uv0 + u * uv1 + v * uv2;
	}
	Vector3 sample_point() const {
		return sample_point(rand(), rand());
	}
	Vector3 sample_point(real x, real y) const {
		if (x + y > 1) {
			x = 1 - x;
			y = 1 - y;
		}
		return v[0] + v10 * x + v20 * y;
	}
	float get_height(Vector3 p) const {
		return dot(normal, p - v[0]);
	}
	int get_relative_location_to_plane(Vector3 p) const {
		return sgn(get_height(p));
	}
	float max_edge_length(int &max_id) const {
		float ret = 0;
		for (int i = 0; i < 3; i++) {
			float dist = length(v[i] - v[(i + 1) % 3]);
			if (dist > ret) {
				max_id = i;
				ret = dist;
			}
		}
		return ret;
	}
};

class Instance {
	Matrix4 transform;
	int id;
};


TC_NAMESPACE_END

