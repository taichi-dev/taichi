// Adapted from https://github.com/PENGUINLIONG/graphi-t

// Copyright (c) 2019 Rendong Liang
//
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

// JSON generated ser/de.
// @PENGUINLIONG
#pragma once
#include <memory>
#include <array>
#include <vector>
#include <map>
#include <unordered_map>
#include <type_traits>
#include <optional>
#include "taichi/common/json.h"

namespace liong {
namespace json {

namespace detail {

struct FieldNameList {
  const std::vector<std::string> field_names;

  static std::vector<std::string> split_field_names(const char *field_names) {
    std::vector<std::string> out{};
    std::string buf{};

    const char *pos = field_names;
    for (char c = *pos; c != '\0'; c = *(++pos)) {
      bool is_lower = (c >= 'a' && c <= 'z');
      bool is_upper = (c >= 'A' && c <= 'Z');
      bool is_digit = (c >= '0' && c <= '9');
      bool is_underscore = c == '_';

      if (is_lower || is_upper || is_digit || is_underscore) {
        buf.push_back(c);
      } else {
        if (!buf.empty()) {
          out.emplace_back(std::exchange(buf, std::string()));
        }
      }
    }
    if (!buf.empty()) {
      out.emplace_back(std::move(buf));
    }
    return out;
  }

  explicit FieldNameList(const char *field_names)
      : field_names(split_field_names(field_names)) {
  }
};

template <typename T>
struct JsonSerde {
  // Numeric and boolean types (integers and floating-point numbers).
  template <typename U = typename std::remove_cv<T>::type>
  static JsonValue serialize(
      typename std::enable_if_t<std::is_arithmetic<U>::value, T> x) {
    return JsonValue(x);
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<std::is_arithmetic<U>::value, T> &x) {
    x = (T)j;
  }
  template <typename U = typename std::remove_cv<T>::type>
  static JsonValue serialize(
      typename std::enable_if_t<std::is_enum<U>::value, T> x) {
    return JsonValue((typename std::underlying_type<T>::type)x);
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<std::is_enum<U>::value, T> &x) {
    x = (T)(typename std::underlying_type<T>::type)j;
  }

  // String type.
  template <typename U = typename std::remove_cv<T>::type>
  static JsonValue serialize(
      typename std::enable_if_t<std::is_same<U, std::string>::value, T> x) {
    return JsonValue(x);
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<std::is_same<U, std::string>::value, T> &x) {
    x = (T)j;
  }

  // Structure types (with a `FieldNameList` field provided).
  template <typename U = typename std::remove_cv<T>::type>
  static JsonValue serialize(
      const typename std::enable_if_t<
          std::is_same<decltype(std::declval<U>().json_serialize_fields()),
                       JsonValue>::value,
          T> &x) {
    return JsonValue(x.json_serialize_fields());
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<
          std::is_same<decltype(std::declval<U>().json_deserialize_fields(
                           std::declval<const JsonObject &>())),
                       void>::value,
          T> &x) {
    x.json_deserialize_fields((const JsonObject &)j);
  }

  // Key-value pairs.
  template <typename U = typename std::remove_cv<T>::type>
  static JsonValue serialize(const typename std::enable_if_t<
                             std::is_same<std::pair<typename U::first_type,
                                                    typename U::second_type>,
                                          T>::value,
                             T> &x) {
    JsonObject obj{};
    obj.inner.emplace(std::make_pair<const std::string, JsonValue>(
        "key", JsonSerde<typename T::first_type>::serialize(x.first)));
    obj.inner.emplace(std::make_pair<const std::string, JsonValue>(
        "value", JsonSerde<typename T::second_type>::serialize(x.second)));
    return JsonValue(std::move(obj));
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<std::is_same<std::pair<typename U::first_type,
                                                       typename U::second_type>,
                                             T>::value,
                                T> &x) {
    JsonSerde<typename T::first_type>::deserialize(j["key"], x.first);
    JsonSerde<typename T::second_type>::deserialize(j["value"], x.second);
  }

  // Owned pointer (requires default constructable).
  template <typename U = typename std::remove_cv<T>::type>
  static JsonValue serialize(
      const typename std::enable_if_t<
          std::is_same<std::unique_ptr<typename U::element_type>, T>::value,
          T> &x) {
    if (x == nullptr) {
      return JsonValue(nullptr);
    } else {
      return JsonSerde<typename T::element_type>::serialize(*x);
    }
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<
          std::is_same<std::unique_ptr<typename U::element_type>, T>::value,
          T> &x) {
    if (j.is_null()) {
      x = nullptr;
    } else {
      x = std::make_unique<typename T::element_type>();
      JsonSerde<typename T::element_type>::deserialize(j, *x);
    }
  }

  // Array types (requires default + move constructable).
  template <typename U = typename std::remove_cv<T>::type>
  static JsonValue serialize(
      const typename std::enable_if_t<std::is_array<U>::value, T> &x) {
    JsonArray arr{};
    for (const auto &xx : x) {
      arr.inner.emplace_back(
          JsonSerde<typename std::remove_extent_t<T>>::serialize(xx));
    }
    return JsonValue(std::move(arr));
  }
  template <typename U = typename std::remove_cv<T>::type>
  static JsonValue serialize(const typename std::enable_if_t<
                             std::is_same<std::array<typename U::value_type,
                                                     std::tuple_size<U>::value>,
                                          T>::value,
                             T> &x) {
    JsonArray arr{};
    for (const auto &xx : x) {
      arr.inner.emplace_back(JsonSerde<typename T::value_type>::serialize(xx));
    }
    return JsonValue(std::move(arr));
  }
  template <typename U = typename std::remove_cv<T>::type>
  static JsonValue serialize(
      const typename std::enable_if_t<
          std::is_same<std::vector<typename U::value_type>, T>::value,
          T> &x) {
    JsonArray arr{};
    for (const auto &xx : x) {
      arr.inner.emplace_back(JsonSerde<typename T::value_type>::serialize(xx));
    }
    return JsonValue(std::move(arr));
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<std::is_array<U>::value, T> &x) {
    for (size_t i = 0; i < std::extent<T>::value; ++i) {
      JsonSerde<typename std::remove_extent_t<T>>::deserialize(j[i], x[i]);
    }
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<
          std::is_same<
              std::array<typename U::value_type, std::tuple_size<U>::value>,
              T>::value,
          T> &x) {
    for (size_t i = 0; i < x.size(); ++i) {
      JsonSerde<typename T::value_type>::deserialize(j[i], x.at(i));
    }
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<
          std::is_same<std::vector<typename U::value_type>, T>::value,
          T> &x) {
    x.clear();
    for (const auto &elem : j.elems()) {
      typename T::value_type xx{};
      JsonSerde<decltype(xx)>::deserialize(elem, xx);
      x.emplace_back(std::move(xx));
    }
  }

  // Dictionary types (requires default + move constructable).
  template <typename U = typename std::remove_cv<T>::type>
  static JsonValue serialize(
      const typename std::enable_if_t<
          std::is_same<std::map<typename U::key_type, typename U::mapped_type>,
                       T>::value,
          T> &x) {
    JsonArray arr{};
    for (const auto &xx : x) {
      arr.inner.emplace_back(JsonSerde<typename T::value_type>::serialize(xx));
    }
    return JsonValue(std::move(arr));
  }
  template <typename U = typename std::remove_cv<T>::type>
  static JsonValue serialize(
      const typename std::enable_if_t<
          std::is_same<
              std::unordered_map<typename U::key_type, typename U::mapped_type>,
              T>::value,
          T> &x) {
    JsonArray arr{};
    for (const auto &xx : x) {
      arr.inner.emplace_back(JsonSerde<typename T::value_type>::serialize(xx));
    }
    return JsonValue(std::move(arr));
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<
          std::is_same<std::map<typename U::key_type, typename U::mapped_type>,
                       T>::value,
          T> &x) {
    x.clear();
    for (const auto &elem : j.elems()) {
      std::pair<typename T::key_type, typename T::mapped_type> xx{};
      JsonSerde<decltype(xx)>::deserialize(elem, xx);
      x.emplace(std::move(*(std::pair<const typename T::key_type,
                                      typename T::mapped_type> *)&xx));
    }
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<
          std::is_same<
              std::unordered_map<typename U::key_type, typename U::mapped_type>,
              T>::value,
          T> &x) {
    x.clear();
    for (const auto &elem : j.elems()) {
      std::pair<typename T::key_type, typename T::mapped_type> xx{};
      JsonSerde<decltype(xx)>::deserialize(elem, xx);
      x.emplace(std::move(*(std::pair<const typename T::key_type,
                                      typename T::mapped_type> *)&xx));
    }
  }

  // Optional types (requires default + move constructable).
  template <typename U = typename std::remove_cv<T>::type>
  static JsonValue serialize(
      const typename std::enable_if_t<
          std::is_same<std::optional<typename U::value_type>, T>::value,
          T> &x) {
    if (x.has_value()) {
      return JsonSerde<typename T::value_type>::serialize(x.value());
    } else {
      return JsonValue(nullptr);
    }
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<
          std::is_same<std::optional<typename U::value_type>, T>::value,
          T> &x) {
    if (j.is_null()) {
      x = std::nullopt;
    } else {
      typename T::value_type xx;
      JsonSerde<typename T::value_type>::deserialize(j, xx);
      x = std::move(xx);
    }
  }
};

template <typename... TArgs>
struct JsonSerdeFieldImpl {};
template <typename TFirst, typename... TOthers>
struct JsonSerdeFieldImpl<TFirst, TOthers...> {
  inline static void serialize(JsonObject &obj,
                               std::vector<std::string>::const_iterator name,
                               const TFirst &first,
                               const TOthers &...others) {
    obj.inner.emplace(std::make_pair<std::string, JsonValue>(
        std::string(*name), JsonSerde<TFirst>::serialize(first)));
    JsonSerdeFieldImpl<TOthers...>::serialize(obj, ++name, others...);
  }
  inline static void deserialize(const JsonObject &obj,
                                 std::vector<std::string>::const_iterator name,
                                 TFirst &first,
                                 TOthers &...others) {
    JsonSerde<TFirst>::deserialize(obj.inner.at(*name), first);
    JsonSerdeFieldImpl<TOthers...>::deserialize(obj, ++name, others...);
  }
};
template <>
struct JsonSerdeFieldImpl<> {
  inline static void serialize(JsonObject &obj,
                               std::vector<std::string>::const_iterator name) {
  }
  inline static void deserialize(
      const JsonObject &obj,
      std::vector<std::string>::const_iterator name) {
  }
};
template <typename... TArgs>
inline void json_serialize_field_impl(
    JsonObject &obj,
    std::vector<std::string>::const_iterator name,
    const TArgs &...args) {
  JsonSerdeFieldImpl<TArgs...>::serialize(obj, name, args...);
}
template <typename... TArgs>
inline void json_deserialize_field_impl(
    const JsonObject &obj,
    std::vector<std::string>::const_iterator name,
    TArgs &...args) {
  JsonSerdeFieldImpl<TArgs...>::deserialize(obj, name, args...);
}

}  // namespace detail

// Serialize a JSON serde object, turning in-memory representations into JSON
// text.
template <typename T>
JsonValue serialize(const T &x) {
  return detail::JsonSerde<T>::serialize(x);
}

// Deserialize a JSON serde object, turning JSON text into in-memory
// representations.
template <typename T>
void deserialize(const JsonValue &j, T &out) {
  detail::JsonSerde<T>::deserialize(j, out);
}

// If you need to control the serialization process on your own, you might want
// to inherit from this.
struct CustomJsonSerdeBase {
 public:
  // Serialize the field values into a JSON object.
  virtual JsonObject json_serialize_fields() const = 0;
  // Deserialize the current object with JSON fields.
  virtual void json_deserialize_fields(const JsonObject &j) = 0;
};

}  // namespace json
}  // namespace liong

#define L_JSON_SERDE_FIELDS(...)                                        \
  const std::vector<std::string> &json_serde_field_names() const {      \
    static ::liong::json::detail::FieldNameList JSON_SERDE_FIELD_NAMES{ \
        #__VA_ARGS__};                                                  \
    return JSON_SERDE_FIELD_NAMES.field_names;                          \
  }                                                                     \
  ::liong::json::JsonValue json_serialize_fields() const {              \
    ::liong::json::JsonObject out{};                                    \
    ::liong::json::detail::json_serialize_field_impl(                   \
        out, json_serde_field_names().begin(), __VA_ARGS__);            \
    return ::liong::json::JsonValue(std::move(out));                    \
  }                                                                     \
  void json_deserialize_fields(const ::liong::json::JsonObject &j) {    \
    ::liong::json::detail::json_deserialize_field_impl(                 \
        j, json_serde_field_names().begin(), __VA_ARGS__);              \
  }
