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

// JSON serialization/deserialization.
// @PENGUINLIONG
#pragma once
#include <string>
#include <vector>
#include <map>
#include <sstream>

namespace liong {
namespace json {

// Any error occured during JSON serialization/deserialization.
class JsonException : public std::exception {
 private:
  std::string msg_;

 public:
  explicit JsonException(std::string_view msg) : msg_(msg) {
  }
  const char *what() const noexcept override {
    return msg_.c_str();
  }
};

// Type of JSON value.
enum JsonType {
  L_JSON_NULL,
  L_JSON_BOOLEAN,
  L_JSON_FLOAT,
  L_JSON_INT,
  L_JSON_STRING,
  L_JSON_OBJECT,
  L_JSON_ARRAY,
};

struct JsonValue;

class JsonElementEnumerator {
  std::vector<JsonValue>::const_iterator beg_, end_;

 public:
  explicit JsonElementEnumerator(const std::vector<JsonValue> &arr)
      : beg_(arr.cbegin()), end_(arr.cend()) {
  }

  std::vector<JsonValue>::const_iterator begin() const {
    return beg_;
  }
  std::vector<JsonValue>::const_iterator end() const {
    return end_;
  }
};

class JsonFieldEnumerator {
  std::map<std::string, JsonValue>::const_iterator beg_, end_;

 public:
  explicit JsonFieldEnumerator(const std::map<std::string, JsonValue> &obj)
      : beg_(obj.cbegin()), end_(obj.cend()) {
  }

  std::map<std::string, JsonValue>::const_iterator begin() const {
    return beg_;
  }
  std::map<std::string, JsonValue>::const_iterator end() const {
    return end_;
  }
};

// JSON array builder.
struct JsonArray {
  std::vector<JsonValue> inner;

  inline JsonArray() = default;
  inline explicit JsonArray(std::vector<JsonValue> &&b) : inner(std::move(b)) {
  }
  inline JsonArray(std::initializer_list<JsonValue> &&elems) : inner(elems) {
  }
};
// JSON object builder.
struct JsonObject {
  std::map<std::string, JsonValue> inner;

  inline JsonObject() = default;
  inline explicit JsonObject(std::map<std::string, JsonValue> &&b)
      : inner(std::move(b)) {
  }
  inline JsonObject(
      std::initializer_list<std::pair<const std::string, JsonValue>> &&fields)
      : inner(fields) {
  }
};

// Represent a abstract value in JSON representation.
struct JsonValue {
  JsonType ty;
  bool b;
  int64_t num_int;
  double num_float;
  std::string str;
  JsonObject obj;
  JsonArray arr;

  inline explicit JsonValue() : ty(L_JSON_NULL) {
  }
  inline explicit JsonValue(std::nullptr_t) : ty(L_JSON_NULL) {
  }
  inline explicit JsonValue(bool b) : ty(L_JSON_BOOLEAN), b(b) {
  }
  inline explicit JsonValue(double num) : ty(L_JSON_FLOAT), num_float(num) {
  }
  inline explicit JsonValue(float num) : ty(L_JSON_FLOAT), num_float(num) {
  }
  inline explicit JsonValue(char num) : ty(L_JSON_INT), num_int(num) {
  }
  inline explicit JsonValue(signed char num) : ty(L_JSON_INT), num_int(num) {
  }
  inline explicit JsonValue(unsigned char num) : ty(L_JSON_INT), num_int(num) {
  }
  inline explicit JsonValue(short num) : ty(L_JSON_INT), num_int(num) {
  }
  inline explicit JsonValue(unsigned short num) : ty(L_JSON_INT), num_int(num) {
  }
  inline explicit JsonValue(int num) : ty(L_JSON_INT), num_int(num) {
  }
  inline explicit JsonValue(unsigned int num) : ty(L_JSON_INT), num_int(num) {
  }
  inline explicit JsonValue(long num) : ty(L_JSON_INT), num_int(num) {
  }
  inline explicit JsonValue(unsigned long num) : ty(L_JSON_INT), num_int(num) {
  }
  inline explicit JsonValue(long long num) : ty(L_JSON_INT), num_int(num) {
  }
  inline explicit JsonValue(unsigned long long num)
      : ty(L_JSON_INT), num_int(num) {
  }
  inline explicit JsonValue(const char *str) : ty(L_JSON_STRING), str(str) {
  }
  inline explicit JsonValue(const std::string &str)
      : ty(L_JSON_STRING), str(str) {
  }
  inline explicit JsonValue(std::string &&str)
      : ty(L_JSON_STRING), str(std::forward<std::string>(str)) {
  }
  inline explicit JsonValue(JsonObject &&obj)
      : ty(L_JSON_OBJECT), obj(std::move(obj.inner)) {
  }
  inline explicit JsonValue(JsonArray &&arr)
      : ty(L_JSON_ARRAY), arr(move(arr.inner)) {
  }

  inline JsonValue &operator[](const char *key) {
    if (!is_obj()) {
      throw JsonException("value is not an object");
    }
    return obj.inner.at(key);
  }
  inline const JsonValue &operator[](const char *key) const {
    if (!is_obj()) {
      throw JsonException("value is not an object");
    }
    return obj.inner.at(key);
  }
  inline JsonValue &operator[](const std::string &key) {
    if (!is_obj()) {
      throw JsonException("value is not an object");
    }
    return obj.inner.at(key);
  }
  inline const JsonValue &operator[](const std::string &key) const {
    if (!is_obj()) {
      throw JsonException("value is not an object");
    }
    return obj.inner.at(key);
  }
  inline JsonValue &operator[](size_t i) {
    if (!is_arr()) {
      throw JsonException("value is not an array");
    }
    return arr.inner.at(i);
  }
  inline const JsonValue &operator[](size_t i) const {
    if (!is_arr()) {
      throw JsonException("value is not an array");
    }
    return arr.inner.at(i);
  }
  inline explicit operator bool() const {
    if (!is_bool()) {
      throw JsonException("value is not a bool");
    }
    return b;
  }
  inline explicit operator double() const {
    if (!is_num()) {
      throw JsonException("value is not a number");
    }
    return num_float;
  }
  inline explicit operator float() const {
    if (!is_num()) {
      throw JsonException("value is not a number");
    }
    return (float)num_float;
  }
  inline explicit operator char() const {
    if (!is_num()) {
      throw JsonException("value is not a number");
    }
    return (char)num_int;
  }
  inline explicit operator signed char() const {
    if (!is_num()) {
      throw JsonException("value is not a number");
    }
    return (signed char)num_int;
  }
  inline explicit operator unsigned char() const {
    if (!is_num()) {
      throw JsonException("value is not a number");
    }
    return (unsigned char)num_int;
  }
  inline explicit operator short() const {
    if (!is_num()) {
      throw JsonException("value is not a number");
    }
    return (short)num_int;
  }
  inline explicit operator unsigned short() const {
    if (!is_num()) {
      throw JsonException("value is not a number");
    }
    return (unsigned short)num_int;
  }
  inline explicit operator int() const {
    if (!is_num()) {
      throw JsonException("value is not a number");
    }
    return (int)num_int;
  }
  inline explicit operator unsigned int() const {
    if (!is_num()) {
      throw JsonException("value is not a number");
    }
    return (unsigned int)num_int;
  }
  inline explicit operator long() const {
    if (!is_num()) {
      throw JsonException("value is not a number");
    }
    return (long)num_int;
  }
  inline explicit operator unsigned long() const {
    if (!is_num()) {
      throw JsonException("value is not a number");
    }
    return (unsigned long)num_int;
  }
  inline explicit operator long long() const {
    if (!is_num()) {
      throw JsonException("value is not a number");
    }
    return (long long)num_int;
  }
  inline explicit operator unsigned long long() const {
    if (!is_num()) {
      throw JsonException("value is not a number");
    }
    return (unsigned long long)num_int;
  }
  inline explicit operator const std::string &() const {
    if (!is_str()) {
      throw JsonException("value is not a string");
    }
    return str;
  }
  inline explicit operator const JsonArray &() const {
    if (!is_arr()) {
      throw JsonException("value is not an array");
    }
    return arr;
  }
  inline explicit operator const JsonObject &() const {
    if (!is_obj()) {
      throw JsonException("value is not an object");
    }
    return obj;
  }

  inline bool is_null() const {
    return ty == L_JSON_NULL;
  }
  inline bool is_bool() const {
    return ty == L_JSON_BOOLEAN;
  }
  inline bool is_num() const {
    return ty == L_JSON_FLOAT || ty == L_JSON_INT;
  }
  inline bool is_str() const {
    return ty == L_JSON_STRING;
  }
  inline bool is_obj() const {
    return ty == L_JSON_OBJECT;
  }
  inline bool is_arr() const {
    return ty == L_JSON_ARRAY;
  }

  inline size_t size() const {
    if (is_obj()) {
      return obj.inner.size();
    } else if (is_arr()) {
      return arr.inner.size();
    } else {
      throw JsonException("only object and array can have size");
    }
  }
  inline JsonElementEnumerator elems() const {
    return JsonElementEnumerator(arr.inner);
  }
  inline JsonFieldEnumerator fields() const {
    return JsonFieldEnumerator(obj.inner);
  }
};

// Parse JSON literal into and `JsonValue` object. If the JSON is invalid or
// unsupported, `JsonException` will be raised.
JsonValue parse(const char *beg, const char *end);
JsonValue parse(const std::string &json_lit);
// Returns true when JSON parsing successfully finished and parsed value is
// returned via `out`. Otherwise, false is returned and out contains incomplete
// result.
bool try_parse(const std::string &json_lit, JsonValue &out);

std::string print(const JsonValue &json);

}  // namespace json
}  // namespace liong
