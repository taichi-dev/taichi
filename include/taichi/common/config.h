#pragma once

#include <map>
#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <typeinfo>
#include <boost/algorithm/string.hpp>

#include <taichi/common/util.h>
#include <taichi/math/math_util.h>
#include <taichi/math/linalg.h>

TC_NAMESPACE_BEGIN

#define LOAD_CONFIG(name, default_val) decltype(default_val) name = config.get(#name, default_val)

class Config {
private:
	std::map<std::string, std::string> data;
	std::vector<std::string> file_names;

public:
	Config() {}

	void print_all() {
		return;
		// We no longer need this??
		std::cout << "Configures: " << std::endl;
		for (auto key = data.begin(); key != data.end(); key++) {
			std::cout << " * " << key->first << " = " << key->second << std::endl;
		}
	}

	float get_float(std::string key) const {
		return (float)std::atof(get_string(key).c_str());
	}

	double get_double(std::string key) const {
		return (double)std::atof(get_string(key).c_str());
	}

	real get_real(std::string key) const {
		return (real)std::atof(get_string(key).c_str());
	}

	int get_int(std::string key) const {
		return std::atoi(get_string(key).c_str());
	}

	int64 get_int64(std::string key) const {
		return std::atoll(get_string(key).c_str());
	}

	unsigned get_unsigned(std::string key) const {
		return unsigned(std::atoll(get_string(key).c_str()));
	}

#define DEFINE_GET(t) t get(std::string key, t default_val) const {if (data.find(key) == data.end()) {return default_val;} else return get_##t(key);}

	DEFINE_GET(int)
		DEFINE_GET(int64)
		DEFINE_GET(unsigned)
		DEFINE_GET(float)
		DEFINE_GET(double)
		DEFINE_GET(bool)
		std::string get(std::string key, const std::string &default_val) const { if (data.find(key) == data.end()) { return default_val; } else return get_string(key); }
	std::string get(std::string key, const char *default_val) const { if (data.find(key) == data.end()) { return default_val; } else return get_string(key); }
	Vector2 get(std::string key, const Vector2 &default_val) const { if (data.find(key) == data.end()) { return default_val; } else return get_vec2(key); }
	Vector3 get(std::string key, const Vector3 &default_val) const { if (data.find(key) == data.end()) { return default_val; } else return get_vec3(key); }
	Vector4 get(std::string key, const Vector4 &default_val) const { if (data.find(key) == data.end()) { return default_val; } else return get_vec4(key); }

	bool has_key(std::string key) const {
		return data.find(key) != data.end();
	}

	std::vector<std::string> get_string_arr(std::string key) const {
		std::string str = get_string(key);
		std::vector<std::string> strs;
		boost::split(strs, str, boost::is_any_of(","));
		for (auto &s : strs) {
			boost::algorithm::trim(s);
		}
		return strs;
	}

	template <typename T>
	T *get_ptr(std::string key) const {
		std::string val = get_string(key);
		std::stringstream ss(val);
		std::string t;
		int64 ptr_ll;
		std::getline(ss, t, '\t');
		ss >> ptr_ll;
		assert_info(t == typeid(T).name(), "Pointer type mismatch: " + t + " and " + typeid(T).name());
		return reinterpret_cast<T*>(ptr_ll);
	}

	bool get_bool(std::string key) const {
		std::string s = get_string(key);
		static std::map<std::string, bool> dict{
			{"true", true},
			{"True", true},
			{"t", true},
			{"1", true},
			{"false", false},
			{"False", false},
			{"f", false},
			{"0", false},
		};
		assert_info(dict.find(s) != dict.end(), "Unkown identifer for bool: " + s);
		return dict[s];
	}

	vec2 get_vec2(std::string key) const {
		vec2 ret;
		sscanf_s(get_string(key).c_str(), "(%f,%f)", &ret.x, &ret.y);
		return ret;
	}

	Vector2i get_vec2i(std::string key) const {
		Vector2i ret;
		sscanf_s(get_string(key).c_str(), "(%d,%d)", &ret.x, &ret.y);
		return ret;
	}

	Vector3 get_vec3(std::string key) const {
		Vector3 ret;
		sscanf_s(get_string(key).c_str(), "(%f,%f,%f)", &ret.x, &ret.y, &ret.z);
		return ret;
	}

	Vector3i get_vec3i(std::string key) const {
		Vector3i ret;
		sscanf_s(get_string(key).c_str(), "(%d,%d,%d)", &ret.x, &ret.y, &ret.z);
		return ret;
	}

	Vector4 get_vec4(std::string key) const {
		Vector4 ret;
		sscanf_s(get_string(key).c_str(), "(%f,%f,%f,%f)", &ret.x, &ret.y, &ret.z, &ret.w);
		return ret;
	}

	Vector4i get_vec4i(std::string key) const {
		Vector4i ret;
		sscanf_s(get_string(key).c_str(), "(%d,%d,%d,%d)", &ret.x, &ret.y, &ret.z, &ret.w);
		return ret;
	}

	template <typename T>
	Config &set(std::string name, T val) {
		std::stringstream ss;
		ss << val;
		data[name] = ss.str();
		return *this;
	}

	Config &set(std::string name, const char *val) {
		std::stringstream ss;
		ss << val;
		data[name] = ss.str();
		return *this;
	}

	Config &set(std::string name, Vector2 val) {
		std::stringstream ss;
		ss << "(" << val.x << "," << val.y << ")";
		data[name] = ss.str();
		return *this;
	}

	Config &set(std::string name, Vector3 val) {
		std::stringstream ss;
		ss << "(" << val.x << "," << val.y << "," << val.z << ")";
		data[name] = ss.str();
		return *this;
	}

	Config &set(std::string name, Vector4 val) {
		std::stringstream ss;
		ss << "(" << val.x << "," << val.y << "," << val.z << "," << val.w << ")";
		data[name] = ss.str();
		return *this;
	}

	Config &set(std::string name, Vector2i val) {
		std::stringstream ss;
		ss << "(" << val.x << "," << val.y << ")";
		data[name] = ss.str();
		return *this;
	}

	Config &set(std::string name, Vector3i val) {
		std::stringstream ss;
		ss << "(" << val.x << "," << val.y << "," << val.z << ")";
		data[name] = ss.str();
		return *this;
	}

	Config &set(std::string name, Vector4i val) {
		std::stringstream ss;
		ss << "(" << val.x << "," << val.y << "," << val.z << "," << val.w << ")";
		data[name] = ss.str();
		return *this;
	}

	template <typename T>
	static std::string get_ptr_string(T *ptr) {
		std::stringstream ss;
		ss << typeid(T).name() << "\t" << reinterpret_cast<uint64>(ptr);
		return ss.str();
	}

	template <typename T>
	Config &set(std::string name, T *ptr) {
		data[name] = get_ptr_string(ptr);
		return *this;
	}

	std::string get_all_file_names() const {
		std::string ret = "";
		for (auto f : file_names)
			ret += f + " ";
		return ret;
	}

	std::string get_string(std::string key) const {
		if (data.find(key) == data.end()) {
			assert_info(false, "No key named '" + key + "' found! [Config files: " + get_all_file_names() + "]");
		}
		return data.find(key)->second;
	}
};

TC_NAMESPACE_END
