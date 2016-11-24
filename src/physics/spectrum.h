#pragma once

#include "math/linalg.h"
#include <vector>

TC_NAMESPACE_BEGIN

class Spectrum {
	std::vector<Vector3d> samples;
	const int maximum_temperature = 20000;
public:
	Spectrum();
	Vector3d sample(real temperature) {
		assert_info(0 <= temperature && temperature < maximum_temperature, "Spectrum Query out of Range");
		int index = (int)floor(temperature);
		double frac = temperature - index;
		return samples[index] * (1 - frac) + samples[index + 1] * (frac);
	}
	static Spectrum &get_instance() {
		static Spectrum spectrum;
		return spectrum;
	}
};

TC_NAMESPACE_END

