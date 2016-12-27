#pragma once 

#include <taichi/common/meta.h>
#include <taichi/common/meta.h>
#include <taichi/math/linalg.h>
#include <taichi/math/math_util.h>
#include <taichi/visual/sampler.h>
#include <taichi/math/discrete_sampler.h>
#include <taichi/visualization/image_buffer.h>
#include <taichi/physics/physics_constants.h>
#include <memory>

TC_NAMESPACE_BEGIN

class EnvironmentMap : public Unit {
public:
	virtual void initialize(const Config &config);

	virtual void set_transform(Matrix4 transform) {
		this->local2world = transform;
		this->world2local = glm::inverse(transform);
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
		return multiply_matrix4(local2world, Vector3(xz * cos(phi), y, xz * sin(phi)), 0);
	}

	Vector2 direction_to_uv(const Vector3 &dir_) const {
		auto dir = multiply_matrix4(world2local, normalized(dir_), 0);
		real theta = std::acos(dir.y);
		real phi = std::atan2(dir.z, dir.x);
		if (phi < 0) {
			phi += 2 * pi;
		}
		return Vector2(phi / (2 * pi), theta / pi);
	}
	real avg_illum;

	Matrix4 local2world;
	Matrix4 world2local;
};

TC_INTERFACE(EnvironmentMap);

TC_NAMESPACE_END

#pragma once
