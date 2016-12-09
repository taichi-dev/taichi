#pragma once

#include "math/linalg.h"
#include <sstream>

TC_NAMESPACE_BEGIN

inline Vector3 load_vector3(std::string dat) {
	Vector3 vec;
	std::stringstream ss(dat);
	for (int i = 0; i < 3; i++)
		ss >> vec[i];
	return vec;
}

TC_NAMESPACE_END

