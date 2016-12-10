#pragma once

#include "math/linalg.h"
#include <sstream>
#include <cstdio>
#include <vector>

TC_NAMESPACE_BEGIN

template <typename T>
void write_vector_to_disk(std::vector<T> *p_vec, std::string fn) {
	std::vector<T> &vec = *p_vec;
	FILE *f = fopen(fn.c_str(), "wb");
	size_t length = vec.size();
	fwrite(&length, sizeof(length), 1, f);
	fwrite(&vec[0], sizeof(vec[0]), length, f);
	fclose(f);
}

template <typename T>
bool read_vector_from_disk(std::vector<T> *p_vec, std::string fn) {
	std::vector<T> &vec = *p_vec;
	P(fn);
	FILE *f = fopen(fn.c_str(), "rb");
	if (f == nullptr) {
		return false;
	}
	size_t length;
	size_t ret = fread(&length, sizeof(length), 1, f);
	P(length);
	if (ret != 1) {
		return false;
	}
	vec.resize(length);
	ret = fread(&vec[0], sizeof(vec[0]), length, f);
	if (ret != length) {
		return false;
	}
	fclose(f);
	return true;
}

TC_NAMESPACE_END

