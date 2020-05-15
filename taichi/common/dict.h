/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <map>
#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <typeinfo>

#include "taichi/common/core.h"
#include "taichi/math/math.h"

TI_NAMESPACE_BEGIN

// Declare and then load
// Load to `this`
#define TI_LOAD_CONFIG(name, default_val) \
  this->name = config.get(#name, default_val)

class Dict {
 private:
  std::map<std::string, std::string> data;

 public:
  TI_IO_DEF(data);

  Dict() = default;

  template <typename T>
  Dict(const std::string &key, const T &value) {
    this->set(key, value);
  }

  std::vector<std::string> get_keys() const {
    std::vector<std::string> keys;
    for (auto it = data.begin(); it != data.end(); ++it) {
      keys.push_back(it->first);
    }
    return keys;
  }

  void clear() {
    data.clear();
  }

  template <typename V>
  typename std::enable_if_t<(!type::is_VectorND<V>() &&
                             !std::is_reference<V>::value &&
                             !std::is_pointer<V>::value),
                            V>
  get(std::string key) const;

  static bool is_string_integral(const std::string &str) {
    // TODO: make it correct
    if (str.find('.') != std::string::npos) {
      return false;
    }
    if (str.find('e') != std::string::npos) {
      return false;
    }
    if (str.find('E') != std::string::npos) {
      return false;
    }
    return true;
  }

  void check_string_integral(const std::string &str) const {
    if (!is_string_integral(str)) {
      TI_ERROR(
          "Getting integral value out of non-integral string '{}' is not "
          "allowed.",
          str);
    }
  }

  void check_value_integral(const std::string &key) const {
    auto str = get_string(key);
    check_string_integral(str);
  }

  template <
      typename V,
      typename std::enable_if<(type::is_VectorND<V>()), V>::type * = nullptr>
  V get(std::string key) const {
    constexpr int N = V::dim;
    using T = typename V::ScalarType;

    std::string str = this->get_string(key);
    std::string temp;
    if (str[0] == '(') {
      temp = "(";
    } else if (str[0] == '[') {
      temp = "[";
    }
    if (std::is_integral<T>()) {
      check_string_integral(str);
    }
    for (int i = 0; i < N; i++) {
      std::string placeholder;
      if (std::is_same<T, float32>()) {
        placeholder = "%f";
      } else if (std::is_same<T, float64>()) {
        placeholder = "%lf";
      } else if (std::is_same<T, int32>()) {
        placeholder = "%d";
      } else if (std::is_same<T, uint32>()) {
        placeholder = "%u";
      } else if (std::is_same<T, int64>()) {
#ifdef WIN32
        placeholder = "%I64d";
#else
        placeholder = "%lld";
#endif
      } else if (std::is_same<T, uint64>()) {
#ifdef WIN32
        placeholder = "%I64u";
#else
        placeholder = "%llu";
#endif
      } else {
        assert(false);
      }
      temp += placeholder;
      if (i != N - 1) {
        temp += ",";
      }
    }
    if (str[0] == '(') {
      temp += ")";
    } else if (str[0] == '[') {
      temp += "]";
    }
    VectorND<N, T> ret;
    if (N == 1) {
      sscanf(str.c_str(), temp.c_str(), &ret[0]);
    } else if (N == 2) {
      sscanf(str.c_str(), temp.c_str(), &ret[0], &ret[1]);
    } else if (N == 3) {
      sscanf(str.c_str(), temp.c_str(), &ret[0], &ret[1], &ret[2]);
    } else if (N == 4) {
      sscanf(str.c_str(), temp.c_str(), &ret[0], &ret[1], &ret[2], &ret[3]);
    }
    return ret;
  }

  std::string get(std::string key, const char *default_val) const;

  template <typename T>
  T get(std::string key, const T &default_val) const;

  bool has_key(std::string key) const {
    return data.find(key) != data.end();
  }

  std::vector<std::string> get_string_arr(std::string key) const {
    std::string str = get_string(key);
    std::vector<std::string> strs = split_string(str, ",");
    for (auto &s : strs) {
      s = trim_string(s);
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
    TI_ASSERT_INFO(t == typeid(T).name(),
                   "Pointer type mismatch: " + t + " and " + typeid(T).name());
    return reinterpret_cast<T *>(ptr_ll);
  }

  template <typename T>
  std::enable_if_t<std::is_pointer<T>::value, std::remove_pointer_t<T>> get(
      std::string key) const {
    return get_ptr<std::remove_pointer_t<T>>(key);
  }

  template <typename T>
  std::enable_if_t<std::is_reference<T>::value, std::remove_reference_t<T>>
      &get(std::string key) const {
    return *get_ptr<std::remove_reference_t<T>>(key);
  }

  template <typename T>
  T *get_ptr(std::string key, T *default_value) const {
    if (has_key(key)) {
      return get_ptr<T>(key);
    } else {
      return default_value;
    }
  }

  template <typename T>
  Dict &set(std::string name, T val) {
    std::stringstream ss;
    ss << val;
    data[name] = ss.str();
    return *this;
  }

  Dict &set(std::string name, const char *val) {
    std::stringstream ss;
    ss << val;
    data[name] = ss.str();
    return *this;
  }

  Dict &set(std::string name, const Vector2 &val) {
    std::stringstream ss;
    ss << "(" << val.x << "," << val.y << ")";
    data[name] = ss.str();
    return *this;
  }

  Dict &set(std::string name, const Vector3 &val) {
    std::stringstream ss;
    ss << "(" << val.x << "," << val.y << "," << val.z << ")";
    data[name] = ss.str();
    return *this;
  }

  Dict &set(std::string name, const Vector4 &val) {
    std::stringstream ss;
    ss << "(" << val.x << "," << val.y << "," << val.z << "," << val.w << ")";
    data[name] = ss.str();
    return *this;
  }

  Dict &set(std::string name, const Vector2i &val) {
    std::stringstream ss;
    ss << "(" << val.x << "," << val.y << ")";
    data[name] = ss.str();
    return *this;
  }

  Dict &set(std::string name, const Vector3i &val) {
    std::stringstream ss;
    ss << "(" << val.x << "," << val.y << "," << val.z << ")";
    data[name] = ss.str();
    return *this;
  }

  Dict &set(std::string name, const Vector4i &val) {
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
  Dict &set(std::string name, T *const ptr) {
    data[name] = get_ptr_string(ptr);
    return *this;
  }

  std::string get_string(std::string key) const {
    if (data.find(key) == data.end()) {
      TI_ERROR("No key named '{}' found.", key);
    }
    return data.find(key)->second;
  }

  template <typename T>
  Dict &operator()(const std::string &key, const T &value) {
    this->set(key, value);
    return *this;
  }
};

template <>
inline std::string Dict::get<std::string>(std::string key) const {
  return get_string(key);
}

template <typename T>
inline T Dict::get(std::string key, const T &default_val) const {
  if (data.find(key) == data.end()) {
    return default_val;
  } else
    return get<T>(key);
}

inline std::string Dict::get(std::string key, const char *default_val) const {
  if (data.find(key) == data.end()) {
    return default_val;
  } else
    return get<std::string>(key);
}

template <>
inline float32 Dict::get<float32>(std::string key) const {
  return (float32)std::atof(get_string(key).c_str());
}

template <>
inline float64 Dict::get<float64>(std::string key) const {
  return (float64)std::atof(get_string(key).c_str());
}

template <>
inline int32 Dict::get<int32>(std::string key) const {
  check_value_integral(key);
  return std::atoi(get_string(key).c_str());
}

template <>
inline uint32 Dict::get<uint32>(std::string key) const {
  check_value_integral(key);
  return uint32(std::atoll(get_string(key).c_str()));
}

template <>
inline int64 Dict::get<int64>(std::string key) const {
  check_value_integral(key);
  return std::atoll(get_string(key).c_str());
}

template <>
inline uint64 Dict::get<uint64>(std::string key) const {
  check_value_integral(key);
  return std::stoull(get_string(key));
}

template <>
inline bool Dict::get<bool>(std::string key) const {
  std::string s = get_string(key);
  static std::map<std::string, bool> dict{
      {"true", true},   {"True", true},   {"t", true},  {"1", true},
      {"false", false}, {"False", false}, {"f", false}, {"0", false},
  };
  TI_ASSERT_INFO(dict.find(s) != dict.end(), "Unkown identifer for bool: " + s);
  return dict[s];
}

using Config = Dict;

TI_NAMESPACE_END
