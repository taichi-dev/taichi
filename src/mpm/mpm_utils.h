#pragma once

#include "visualization/point_renderer.h"
#include "visualization/texture_renderer.h"
#include <memory>
#include "system/timer.h"
#include "common/config.h"
#include "math/linalg.h"

TC_NAMESPACE_BEGIN

#define Pp(v) {printf("%s:\n", #v); print(v);}
#define abnormal(v) (!is_normal(v))

// Note: assuming abs(x) <= 2!!
inline float w(float x) {
	x = abs(x);
	assert(x <= 2);
	if (x < 1) {
		return 0.5f * x * x * x - x * x + 2.0f / 3.0f;
	}
	else {
		return -1.0f / 6.0f * x * x * x + x * x - 2 * x + 4.0f / 3.0f;
	}
}

// Note: assuming abs(x) <= 2!!
inline float dw(float x) {
	float s = x < 0.0f ? -1.0f : 1.0f;
	x *= s;
	assert(x <= 2.0f);
	float val;
	float xx = x * x;
	if (x < 1.0f) {
		val = 1.5f * xx - 2.0f * x;
	}
	else {
		val = -0.5f * xx + 2.0f * x - 2.0f;
	}
	return s * val;
}

inline float w(const vec2 &a) {
	return w(a.x) * w(a.y);
}

inline vec2 dw(const vec2 &a) {
	return vec2(dw(a.x) * w(a.y), w(a.x) * dw(a.y));
}

inline float det(const mat2 &m) {
	return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

void polar_decomp(const mat2 &A, mat2 &r, mat2 &s);

void svd(const mat2 &A, mat2 &u, mat2 &sig, mat2 &v);


TC_NAMESPACE_END

