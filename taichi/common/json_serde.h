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

namespace type {
template <typename T>
using remove_cvref =
    typename std::remove_cv<typename std::remove_reference<T>::type>;

template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;
}  // namespace type

template <typename T>
struct has_ptr_serde {
  template <typename T_>
  static constexpr auto helper(T_ *) -> std::is_same<
      decltype((T_::jsonserde_ptr_io(std::declval<const T_ *&>(),
                                     std::declval<JsonValue &>(),
                                     std::declval<bool>(),
                                     std::declval<bool>()))),
      void>;

  template <typename>
  static constexpr auto helper(...) -> std::false_type;

 public:
  using T__ = typename type::remove_cvref_t<T>;
  using type = decltype(helper<T__>(nullptr));
  static constexpr bool value = type::value;
};

template <typename T>
struct JsonSerde {
  template <typename T__, typename T_ = typename type::remove_cvref_t<T__>>
  static T_ &get_writable(T__ &&t) {
    return *const_cast<T_ *>(&t);
  }

  // Pointer with a custom serialization function.
  template <typename U = typename std::remove_cv<T>::type>
  static JsonValue serialize(const typename std::enable_if_t<
                             std::is_pointer_v<U> &&
                                 has_ptr_serde<std::remove_pointer_t<T>>::value,
                             T> &x) {
    JsonValue val;
    using T_ = std::remove_pointer_t<T>;
    // NOTE: strict is not used if writing is true
    T_::jsonserde_ptr_io((const T_ *&)x, val, /*writing=*/true,
                         /*strict=*/false);
    return val;
  }

  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<
          std::is_pointer_v<U> &&
              has_ptr_serde<std::remove_pointer_t<U>>::value,
          T> &x,
      bool strict) {
    using T_ = std::remove_pointer_t<T>;
    T_::jsonserde_ptr_io((const T_ *&)x, get_writable(j), /*writing=*/false,
                         /*strict=*/strict);
  }

  // Numeric and boolean types (integers and floating-point numbers).
  template <typename U = typename std::remove_cv<T>::type>
  static JsonValue serialize(
      typename std::enable_if_t<std::is_arithmetic<U>::value, T> x) {
    return JsonValue(x);
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<std::is_arithmetic<U>::value, T> &x,
      bool strict) {
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
      typename std::enable_if_t<std::is_enum<U>::value, T> &x,
      bool strict) {
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
      typename std::enable_if_t<std::is_same<U, std::string>::value, T> &x,
      bool strict) {
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
                           std::declval<const JsonObject &>(),
                           std::declval<bool>())),
                       void>::value,
          T> &x,
      bool strict) {
    x.json_deserialize_fields((const JsonObject &)j, strict);
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
                                T> &x,
      bool strict) {
    JsonSerde<typename T::first_type>::deserialize(j["key"], x.first, strict);
    JsonSerde<typename T::second_type>::deserialize(j["value"], x.second,
                                                    strict);
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
          T> &x,
      bool strict) {
    if (j.is_null()) {
      x = nullptr;
    } else {
      x = std::make_unique<typename T::element_type>();
      JsonSerde<typename T::element_type>::deserialize(j, *x, strict);
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
      typename std::enable_if_t<std::is_array<U>::value, T> &x,
      bool strict) {
    for (size_t i = 0; i < std::extent<T>::value; ++i) {
      JsonSerde<typename std::remove_extent_t<T>>::deserialize(j[i], x[i],
                                                               strict);
    }
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<
          std::is_same<
              std::array<typename U::value_type, std::tuple_size<U>::value>,
              T>::value,
          T> &x,
      bool strict) {
    for (size_t i = 0; i < x.size(); ++i) {
      JsonSerde<typename T::value_type>::deserialize(j[i], x.at(i), strict);
    }
  }
  template <typename U = typename std::remove_cv<T>::type>
  static void deserialize(
      const JsonValue &j,
      typename std::enable_if_t<
          std::is_same<std::vector<typename U::value_type>, T>::value,
          T> &x,
      bool strict) {
    x.clear();
    for (const auto &elem : j.elems()) {
      typename T::value_type xx{};
      JsonSerde<decltype(xx)>::deserialize(elem, xx, strict);
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
          T> &x,
      bool strict) {
    x.clear();
    for (const auto &elem : j.elems()) {
      std::pair<typename T::key_type, typename T::mapped_type> xx{};
      JsonSerde<decltype(xx)>::deserialize(elem, xx, strict);
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
          T> &x,
      bool strict) {
    x.clear();
    for (const auto &elem : j.elems()) {
      std::pair<typename T::key_type, typename T::mapped_type> xx{};
      JsonSerde<decltype(xx)>::deserialize(elem, xx, strict);
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
          T> &x,
      bool strict) {
    if (j.is_null()) {
      x = std::nullopt;
    } else {
      typename T::value_type xx;
      JsonSerde<typename T::value_type>::deserialize(j, xx, strict);
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
                                 bool strict,
                                 std::vector<std::string>::const_iterator name,
                                 TFirst &first,
                                 TOthers &...others) {
    auto it = obj.inner.find(*name);
    if (it != obj.inner.end()) {
      JsonSerde<TFirst>::deserialize(it->second, first, strict);
    } else if (strict) {
      throw ::liong::json::JsonException("Missing field: " + *name);
    }
    JsonSerdeFieldImpl<TOthers...>::deserialize(obj, strict, ++name, others...);
  }
};
template <>
struct JsonSerdeFieldImpl<> {
  inline static void serialize(JsonObject &obj,
                               std::vector<std::string>::const_iterator name) {
  }
  inline static bool deserialize(
      const JsonObject &obj,
      bool strict,
      std::vector<std::string>::const_iterator name) {
    return true;
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
    bool strict,
    std::vector<std::string>::const_iterator name,
    TArgs &...args) {
  if (strict && obj.inner.size() != sizeof...(TArgs)) {
    throw ::liong::json::JsonException("unexpected number of fields");
  }
  return JsonSerdeFieldImpl<TArgs...>::deserialize(obj, strict, name, args...);
}

}  // namespace detail

// Serialize a JSON serde object, turning in-memory representations into JSON
// text.
template <typename T>
JsonValue serialize(const T &x) {
  return detail::JsonSerde<T>::serialize(x);
}

// Deserialize a JSON serde object, turning JSON text into in-memory
// representations. If `strict` is true, the function will throw JsonException
// if a field is missing or an extra field is present. Otherwise, the missing
// fields will be filled with default values and the extra fields will be
// ignored. See serialize_test.cpp for examples.
template <typename T>
void deserialize(const JsonValue &j, T &out, bool strict = false) {
  detail::JsonSerde<T>::deserialize(j, out, strict);
}

// If you need to control the serialization process on your own, you might want
// to inherit from this.
struct CustomJsonSerdeBase {
 public:
  // Serialize the field values into a JSON object.
  virtual JsonObject json_serialize_fields() const = 0;
  // Deserialize the current object with JSON fields.
  virtual void json_deserialize_fields(const JsonObject &j, bool strict) = 0;
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
  void json_deserialize_fields(const ::liong::json::JsonObject &j,      \
                               bool strict) {                           \
    ::liong::json::detail::json_deserialize_field_impl(                 \
        j, strict, json_serde_field_names().begin(), __VA_ARGS__);      \
  }
