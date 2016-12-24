#pragma once

#include <taichi/common/util.h>
#include <taichi/math/linalg.h>
#include <vector>

TC_NAMESPACE_BEGIN

class Ray {
public:
	Ray() {};
	Ray(Vector3 orig, Vector3 dir, real time = 0) : orig(orig),
		dir(dir), dist(DIST_INFINITE), time(time) {
		triangle_id = -1;
	}

	Vector3 orig, dir;
	real time, dist;
	int triangle_id;
	Vector3 geometry_normal;
	real u, v;

	const static real DIST_INFINITE;
};

struct Face {
    Face() { }

    Face(int v0, int v1, int v2) {
        vert_ind[0] = v0;
        vert_ind[1] = v1;
        vert_ind[2] = v2;
    }

    int vert_ind[3];
    int material;
};

struct Triangle {
	Triangle() {};
	int id;
	Vector3 v[3];
	Vector3 v10, v20, iv10, iv20; // iv10 = (v1 - v0) / length(v1 - v0)^2
	Vector3 normal;
	Vector3 n0;
	Vector2 uv0, uv10, uv20;
	Vector3 n10, n20;
	real area;
	real temperature;
	real heat_capacity;
	Triangle(const Vector3 &v0, const Vector3 &v1, const Vector3 &v2,
		const Vector3 &n0, const Vector3 &n1, const Vector3 &n2,
		const Vector2 &uv0, const Vector2 &uv1, const Vector2 &uv2, int id=-1) {
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
		this->n0 = n0;
		this->uv0 = uv0;
		this->uv10 = uv1 - uv0;
		this->uv20 = uv2 - uv0;
		v10 = v1 - v0;
		v20 = v2 - v0;
        iv10 = 1.0f / dot(v10, v10) * v10;
		iv20 = 1.0f / dot(v20, v20) * v20;
		n10 = n1 - n0;
		n20 = n2 - n0;
		this->id = id;
		normal = normalized(cross(v10, v20));
		area = 0.5f * length(cross(v[1] - v[0], v[2] - v[0]));
	}

	Triangle get_transformed(const Matrix4 &transform) const {
		return Triangle(
			multiply_matrix4(transform, v[0], 1.0f),
			multiply_matrix4(transform, v[0] + v10, 1.0f),
			multiply_matrix4(transform, v[0] + v20, 1.0f),
			multiply_matrix4(transform, n0, 0.0f),
			multiply_matrix4(transform, n0 + n10, 0.0f),
			multiply_matrix4(transform, n0 + n20, 0.0f),
			uv0,
			uv0 + uv10,
			uv0 + uv20,
			id
		);
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
		real dir_n = dot(dir, normal);
		real dist_n = dot(v[0] - orig, normal);
		real dist = dist_n / dir_n;
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
		return uv0 + u * uv10 + v * uv20;
	}
	Vector2 get_duv(Vector3 dx) const {
		return uv10 * dot(iv10, dx) + uv20 * dot(iv20, dx);
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
	real get_height(Vector3 p) const {
		return dot(normal, p - v[0]);
	}
	int get_relative_location_to_plane(Vector3 p) const {
		return sgn(get_height(p));
	}
	real max_edge_length(int &max_id) const {
		real ret = 0;
		for (int i = 0; i < 3; i++) {
			real dist = length(v[i] - v[(i + 1) % 3]);
			if (dist > ret) {
				max_id = i;
				ret = dist;
			}
		}
		return ret;
	}
    bool operator == (const Triangle &b) const {
		return false;
	}
};

class Instance {
	Matrix4 transform;
	int id;
};


TC_NAMESPACE_END

