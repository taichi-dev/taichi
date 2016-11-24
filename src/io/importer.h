#pragma once

#include "math/linalg.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <sstream>

TC_NAMESPACE_BEGIN

using boost::property_tree::ptree;


inline Matrix4 load_matrix4(ptree &pt) {
	Matrix4 m;
	int j = 0;
	foreach(ptree::value_type &v, pt) {
		std::stringstream ss(v.second.data());
		for (int i = 0; i < 4; i++) {
			ss >> m[i][j];
		}
		j++;
	}
	return m;
}

inline Vector3 load_vector3(string dat) {
	Vector3 vec;
	std::stringstream ss(dat);
	for (int i = 0; i < 3; i++)
		ss >> vec[i];
	return vec;
}



TC_NAMESPACE_END

