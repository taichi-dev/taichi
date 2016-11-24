#pragma once
#include <map>
#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <boost/algorithm/string.hpp>

#include "utils.h"
#include "math/math_utils.h"
#include "math/linalg.h"

TC_NAMESPACE_BEGIN

using std::string;
using std::map;

#define LOAD_CONFIG(name, default_val) decltype(default_val) name = config.get(#name, default_val)

class Config {
private:
	std::map<std::string, std::string> data;
	std::vector<std::string> file_names;

public:
	Config() {}

	Config &append(std::string config_name) {
		file_names.push_back(config_name);
		// printf("Loading config %s...\n", config_name.c_str());
		std::string config_path = std::string("config/") + config_name;
		std::ifstream fs;
		fs.open(file_search_root + config_path, std::ios::in);
		assert_info(fs.is_open(), std::string("Error: config file [") + config_path.c_str() + "] not found!\n");
		while (true) {
			if (fs.eof()) {
				break;
			}
			std::string line, key, val;
			std::getline(fs, line);
			if (line.size() <= 1) {
				continue;
			}
			int line_length = (int)line.size();
			int i;
			for (i = 0; i <= line_length; i++) {
				if (i == line_length) {
					assert_info(false, "Illegible Line: " + std::string(line));
				}
				if (line[i] == ' ' || line[i] == '\t') {
					break;
				}
				key += line[i];
			}
			line = line.substr(i, (int)line.size() - i);
			while (!line.empty()) {
				if (!(line.front() == ' ' || line.front() == '\t')) {
					break;
				}
				line.erase(line.begin());
			}
			val = line;
			data[key] = val;
		}
		fs.close();
		return *this;
	}

	void print_all() {
		cout << "Configures: " << endl;
		for (auto key = data.begin(); key != data.end(); key++) {
			cout << " * " << key->first << " = " << key->second << endl;
		}
	}

	float get_float(std::string key) const {
		return (float)::atof(get_string(key).c_str());
	}

	double get_double(std::string key) const {
		return (double)::atof(get_string(key).c_str());
	}

	real get_real(std::string key) const {
		return (real)::atof(get_string(key).c_str());
	}

	int get_int(std::string key) const {
		return ::atoi(get_string(key).c_str());
	}

//#define DEFINE_GET(t) template <> t get<t>(std::string key, t default_val) const {if (data.find(key) == data.end()) {return default_val;} else return get_##t(key);}
#define DEFINE_GET(t) t get(std::string key, t default_val) const {if (data.find(key) == data.end()) {return default_val;} else return get_##t(key);}

	DEFINE_GET(int)
	DEFINE_GET(real)
	DEFINE_GET(double)
	DEFINE_GET(bool)
	DEFINE_GET(string)
	std::string get(std::string key, const char *default_val) const {if (data.find(key) == data.end()) {return default_val;} else return get_string(key);}
	Vector2 get(std::string key, const Vector2 &default_val) const {if (data.find(key) == data.end()) {return default_val;} else return get_vec2(key);}
	Vector3 get(std::string key, const Vector3 &default_val) const {if (data.find(key) == data.end()) {return default_val;} else return get_vec3(key);}
	Vector4 get(std::string key, const Vector4 &default_val) const {if (data.find(key) == data.end()) {return default_val;} else return get_vec4(key);}

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

	bool get_bool(std::string key) const {
		string s = get_string(key);
		static map<std::string, bool> dict{
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

	vec3 get_vec3(std::string key) const {
		vec3 ret;
		sscanf_s(get_string(key).c_str(), "(%f,%f,%f)", &ret.x, &ret.y, &ret.z);
		return ret;
	}

	vec4 get_vec4(std::string key) const {
		vec4 ret;
		sscanf_s(get_string(key).c_str(), "(%f,%f,%f,%f)", &ret.x, &ret.y, &ret.z, &ret.w);
		return ret;
	}

	template <typename T>
	Config &set(std::string name, T val) {
		std::stringstream ss;
		ss << val;
		data[name] = ss.str();
		return *this;
	}

	Config &set(std::string name, Vector3 val) {
		std::stringstream ss;
		ss << "(" << val.x << "," << val.y << "," << val.z << ")";
		data[name] = ss.str();
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

	static Config load(std::string config_fn) {
		Config config;
		config.append(config_fn);
		return config;
	}
};

TC_NAMESPACE_END
