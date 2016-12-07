#pragma once

#include "sampler.h"
#include "math/linalg.h"
#include "common/meta.h"
#include "sampler.h"
#include "geometry_primitives.h"

TC_NAMESPACE_BEGIN

enum class VolumeEvent {
	scattering,
	absorption
};

class VolumeMaterial {
protected:
	Matrix4 world2local;
	Matrix4 local2world;
public:
	VolumeMaterial() { 
		set_transform(Matrix4(1.0f)); 
	}
	virtual void initialize(const Config &config) {
		printf("Info: Volumetric rendering is turned ON. Note that PT & MCMCPT are only renderers that support this.\n");
		printf("      This may lead to different output by different renderers.\n");
		this->volumetric_scattering = config.get_real("scattering");
		this->volumetric_absorption = config.get_real("absorption");
	}

	void set_transform(const Matrix4 &local2world) {
		this->local2world = local2world;
		this->world2local = glm::inverse(local2world);
	}

	virtual real sample_free_distance(StateSequence &rand, const Ray &ray) const {
		real kill = volumetric_scattering + volumetric_absorption;
		if (kill > 0) {
			return -log(1 - rand()) / kill;
		}
		else {
			return std::numeric_limits<real>::infinity();
		}
	}

	virtual real unbiased_sample_attenuation(const Vector3 &start, const Vector3 &end, StateSequence &rand) const {
		return get_attenuation(glm::length(multiply_matrix4(world2local, end - start, 0)));
	}

	virtual VolumeEvent sample_event(StateSequence &rand, const Ray &ray) const {
		return rand() < volumetric_scattering / (volumetric_scattering + volumetric_absorption) ?
			VolumeEvent::scattering : VolumeEvent::absorption;
	}

	virtual Vector3 sample_phase(StateSequence &rand, const Ray &ray) const {
		return sample_sphere(rand(), rand());
	}

protected:
	virtual real get_attenuation(real dist) const {
		return exp(-dist * (volumetric_scattering + volumetric_absorption));
	}

	real volumetric_scattering;
	real volumetric_absorption;
};

TC_INTERFACE(VolumeMaterial);

class VolumeStack {
	std::vector<VolumeMaterial const*> stack;
public:
	VolumeStack();
	void push(VolumeMaterial const*vol) {
		stack.push_back(vol);
	}
	void pop() {
		stack.pop_back();
	}
	VolumeMaterial const* top() {
		return stack.back();
	}

	size_t size() const {
		return stack.size();
	}
};

class VolumeStackPushGuard {
private:
	VolumeStack &stack;
public:
	VolumeStackPushGuard(VolumeStack &stack, const VolumeMaterial &volume) : stack(stack) {
		stack.push(&volume);
	}
	~VolumeStackPushGuard() {
		stack.pop();
	}
};
TC_NAMESPACE_END

