#pragma once 

#include "common/config.h"
#include "common/meta.h"
#include "math/linalg.h"
#include "math/math_utils.h"
#include "sampler.h"
#include "discrete_sampler.h"
#include "visualization/image_buffer.h"
#include "physics/physics_constants.h"
#include <memory>

TC_NAMESPACE_BEGIN

class EnvironmentMap {
public:
	virtual void initialize(const Config &config);

	virtual void set_transform(Matrix4 transform) {
		this->transform = transform;
	}

	virtual Vector3 sample_direction(StateSequence &rand, real &pdf, Vector3 &illum) const;
	
	virtual real pdf(const Vector3 &dir) const;

	Vector3 sample_illum(const Vector3 &direction) const {
		return sample_illum(direction_to_uv(direction));
	}

	Vector3 sample_illum(const Vector2 &uv) const {
		return image->sample_relative_coord(uv);
	}

protected:
	std::shared_ptr<ImageBuffer<Vector3>> image;
	int width, height;
	DiscreteSampler row_sampler;
	std::vector<DiscreteSampler> col_samplers;
	void build_cdfs();


	Vector3 uv_to_direction(const Vector2 &uv) const {
		real theta = uv.y * pi;
		real phi = uv.x * 2 * pi;
		real y = cos(theta);
		real xz = sqrt(1 - y * y);
		return Vector3(xz * cos(phi), y, xz * sin(phi));
	}

	Vector2 direction_to_uv(const Vector3 &dir_) const {
		auto dir = normalized(dir_);
		real theta = std::acos(dir.y);
		real phi = std::atan2(dir.z, dir.x);
		if (phi < 0) {
			phi += 2 * pi;
		}
		return Vector2(phi / (2 * pi), theta / pi);
	}
	Matrix4 transform;
	real avg_illum;
};

TC_INTERFACE(EnvironmentMap);

TC_NAMESPACE_END

#pragma once
