/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <map>
#include <fstream>
#include <sstream>

TC_NAMESPACE_BEGIN

////////////////////////////////////////////////////////////////////////////////
//                   A Minimalist Serializer for Taichi                       //
////////////////////////////////////////////////////////////////////////////////

#define TC_IO_DECL      \
  template <typename S> \
  void io(S &serializer) const

#define TC_IO_DEF(...)     \
  template <typename S>    \
  void io(S &serializer) { \
    TC_IO(__VA_ARGS__)     \
  }

/*
#define TC_IO(...) \
  { serializer(#__VA_ARGS__, __VA_ARGS__); }
*/

#define TC_IO(x)                                                            \
  {                                                                         \
    serializer(                                                             \
        #x,                                                                 \
        const_cast<typename std::decay<decltype(*this)>::type *>(this)->x); \
  }

#define TC_SERIALIZER_IS(T)                                                 \
  (std::is_same<typename std::remove_reference<decltype(serializer)>::type, \
                T>())

static_assert(
    sizeof(std::size_t) == sizeof(uint64_t),
    "sizeof(std::size_t) should be 8. Try compiling with 64bit mode.");

template <typename option, typename... Args>
struct type_switch {
  using type = typename std::conditional<
      std::is_same<typename option::first_type, std::true_type>::value,
      typename option::second_type,
      typename type_switch<Args...>::type>::type;
};

template <typename option>
struct type_switch<option> {
  static_assert(
      std::is_same<typename option::first_type, std::true_type>::value,
      "None of the options in type_switch works.");
  using type = typename option::second_type;
};

template <typename... Args>
using type_switch_t = typename type_switch<Args...>::type;

class Serializer {
 public:
  template <typename T, std::size_t n>
  using TArray = T[n];

  template <typename T>
  struct has_io {
    template <typename T_>
    static constexpr auto helper(T_ *)
        -> std::is_same<decltype((std::declval<T_>().template io(
                            std::declval<Serializer &>()))),
                        void>;

    template <typename>
    static constexpr auto helper(...) -> std::false_type;

   public:
    using T__ = typename std::decay<T>::type;
    using type = decltype(helper<T__>(nullptr));
    static constexpr bool value = type::value;
  };

 public:
  template <typename T>
  struct Item {
    using is_array =
        typename std::is_array<typename std::remove_cv<T>::type>::type;
    using is_lref = typename std::is_lvalue_reference<T>::type;

    static_assert(!std::is_pointer<T>(), "");

    using ValueType = type_switch_t<
        std::pair<is_lref, T>,  // Keep l-value references
        std::pair<is_array,
                  typename std::remove_cv<T>::type>,  // Do nothing for arrays
        std::pair<std::true_type, typename std::decay<T>::type>
        // copy r-value references?
        // is there a better way?
        >;

    Item(const std::string &key, ValueType &&value)
        : key(key), value(std::forward<ValueType>(value)) {
    }

    ValueType value;
    const std::string &key;
  };

  template <typename T>
  auto make_item(const std::string &name, T &&t) -> Item<T> {
    return Item<T>(name, std::forward<T>(t));
  }

  template <typename T>
  auto make_item(T &&t) -> Item<T> {
    return Item<T>("", std::forward<T>(t));
  }
};

static_assert(
    std::is_same<typename Serializer::Item<int &>::is_array, std::false_type>(),
    "");

static_assert(
    std::is_same<typename Serializer::Item<int &>::ValueType, int &>(),
    "");

static_assert(std::is_same<typename Serializer::Item<int &&>::ValueType, int>(),
              "");

static_assert(std::is_same<typename Serializer::Item<int &&>::ValueType, int>(),
              "");

static_assert(
    std::is_same<typename Serializer::Item<int[32]>::ValueType, int[32]>(),
    "");

template <bool writing>
class BinarySerializer : public Serializer {
 public:
  std::vector<uint8_t> data;
  uint8_t *c_data;

  std::size_t head;
  std::size_t preserved;

  using Base = Serializer;
  using Base::Item;

  void write_to_file(const std::string &file_name) {
    FILE *f = fopen(file_name.c_str(), "wb");
    assert(f != nullptr);
    void *ptr = c_data;
    if (!ptr) {
      assert(!data.empty());
      ptr = &data[0];
    }
    fwrite(ptr, sizeof(uint8_t), head, f);
    fclose(f);
  }

  template <bool writing_ = writing>
  typename std::enable_if<!writing_, void>::type initialize(
      const std::string &file_name) {
    FILE *f = fopen(file_name.c_str(), "rb");
    assert(f != nullptr);
    std::size_t length = 0;
    while (true) {
      int limit = 1 << 8;
      data.resize(data.size() + limit);
      void *ptr = reinterpret_cast<void *>(&data[length]);
      size_t length_tmp = fread(ptr, sizeof(uint8_t), limit, f);
      length += length_tmp;
      if (length_tmp < limit) {
        break;
      }
    }
    fclose(f);
    data.resize(length);
    c_data = reinterpret_cast<uint8_t *>(&data[0]);
    head = sizeof(std::size_t);
  }

  template <bool writing_ = writing>
  typename std::enable_if<writing_, void>::type initialize(
      std::size_t preserved_ = std::size_t(0),
      void *c_data = nullptr) {
    std::size_t n = 0;
    head = 0;
    if (preserved_ != 0) {
      TC_TRACE("perserved = {}", preserved_);
      // Preserved mode
      this->preserved = preserved_;
      assert(c_data != nullptr);
      this->c_data = (uint8_t *)c_data;
    } else {
      // otherwise use a std::vector<uint8_t>
      this->preserved = 0;
      this->c_data = nullptr;
    }
    this->operator()("", n);
  }

  template <bool writing_ = writing>
  typename std::enable_if<!writing_, void>::type initialize(
      void *raw_data,
      std::size_t preserved_ = std::size_t(0)) {
    if (preserved_ != 0) {
      assert(raw_data == nullptr);
      data.resize(preserved_);
      c_data = &data[0];
    } else {
      assert(raw_data != nullptr);
      c_data = reinterpret_cast<uint8_t *>(raw_data);
    }
    head = sizeof(std::size_t);
    preserved = 0;
  }

  void finalize() {
    if (writing) {
      if (c_data) {
        *reinterpret_cast<std::size_t *>(&c_data[0]) = head;
      } else {
        *reinterpret_cast<std::size_t *>(&data[0]) = head;
      }
    } else {
      assert(head == *reinterpret_cast<std::size_t *>(c_data));
    }
  }

  void operator()(const char *, std::string &val) {
    if (writing) {
      std::vector<char> val_vector(val.begin(), val.end());
      this->operator()(nullptr, val_vector);
    } else {
      std::vector<char> val_vector;
      this->operator()(nullptr, val_vector);
      val = std::string(val_vector.begin(), val_vector.end());
    }
  }

  // C-array
  template <typename T, std::size_t n>
  void operator()(const char *, TArray<T, n> &val) {
    if (writing) {
      for (std::size_t i = 0; i < n; i++) {
        this->operator()("", val[i]);
      }
    } else {
      // TODO: why do I have to let it write to tmp, otherwise I get Sig Fault?
      TArray<T, n> tmp;
      for (std::size_t i = 0; i < n; i++) {
        this->operator()("", tmp[i]);
      }
      std::memcpy(val, tmp, sizeof(tmp));
    }
  }

  // Elementary data types
  template <typename T>
  typename std::enable_if<!has_io<T>::value, void>::type operator()(
      const char *,
      T &&val) {
    if (writing) {
      std::size_t new_size = head + sizeof(T);
      if (c_data) {
        if (new_size > preserved) {
          TC_CRITICAL("Preserved Buffer (size {}) Overflow.", preserved);
        }
        *reinterpret_cast<typename std::remove_reference<T>::type *>(
            &c_data[head]) = val;
      } else {
        data.resize(new_size);
        *reinterpret_cast<typename std::remove_reference<T>::type *>(
            &data[head]) = val;
      }
    } else {
      val = *reinterpret_cast<typename std::remove_reference<T>::type *>(
          &c_data[head]);
    }
    head += sizeof(T);
  }

  template <typename T>
  typename std::enable_if<has_io<T>::value, void>::type operator()(const char *,
                                                                   T &&val) {
    val.io(*this);
  }

  template <typename T>
  void operator()(const char *, std::vector<T> &val) {
    if (writing) {
      this->operator()("", val.size());
    } else {
      std::size_t n;
      this->operator()("", n);
      val.resize(n);
    }
    for (std::size_t i = 0; i < val.size(); i++) {
      this->operator()("", std::forward<T>(val[i]));
    }
  }

  template <typename T, typename G>
  void operator()(const char *, std::pair<T, G> &val) {
    this->operator()(nullptr, val.first);
    this->operator()(nullptr, val.second);
  }

  template <typename T, typename G>
  void operator()(const char *, std::map<T, G> &val) {
    if (writing) {
      this->operator()(nullptr, val.size());
      for (auto iter : val) {
        T first = iter.first;
        this->operator()(nullptr, first);
        this->operator()(nullptr, iter.second);
      }
    } else {
      val.clear();
      std::size_t n;
      this->operator()(nullptr, n);
      for (std::size_t i = 0; i < n; i++) {
        std::pair<T, G> record;
        this->operator()(nullptr, record);
        val.insert(record);
      }
    }
  }

  template <typename T, typename... Args>
  void operator()(const char *, T &&t, Args &&... rest) {
    this->operator()(nullptr, std::forward<T>(t));
    this->operator()(nullptr, std::forward<Args>(rest)...);
  }

  template <typename T>
  void operator()(T &&val) {
    this->operator()(nullptr, std::forward<T>(val));
  }
};

using BinaryOutputSerializer = BinarySerializer<true>;
using BinaryInputSerializer = BinarySerializer<false>;

class TextSerializer : public Serializer {
 public:
  std::string data;
  void print() const {
    std::cout << data << std::endl;
  }

  void write_to_file(const std::string &file_name) {
    std::ofstream fs(file_name);
    fs << data;
    fs.close();
  }

 private:
  int indent;
  static constexpr int indent_width = 2;
  bool first_line;

  void add_line(const std::string &str) {
    if (first_line) {
      first_line = false;
    } else {
      data += "\n";
    }
    data += std::string(indent_width * indent, ' ') + str;
  }

  void add_line(const std::string &key, const std::string &value) {
    add_line(key + ": " + value);
  }

 public:
  TextSerializer() {
    indent = 0;
    first_line = false;
  }

  template <typename T>
  static std::string serialize(const char *key, T &&t) {
    TextSerializer ser;
    ser(key, std::forward<T>(t));
    return ser.data;
  }

  void operator()(const char *key, std::string &val) {
    add_line(std::string(key) + ": " + val);
  }

  template <typename T, std::size_t n>
  using is_compact =
      typename std::integral_constant<bool,
                                      std::is_arithmetic<T>::value && (n < 7)>;

  // C-array
  template <typename T, std::size_t n>
  typename std::enable_if<is_compact<T, n>::value, void>::type operator()(
      const char *key,
      TArray<T, n> &val) {
    std::stringstream ss;
    ss << "[";
    for (std::size_t i = 0; i < n; i++) {
      ss << val[i];
      if (i != n - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    add_line(key, ss.str());
  }

  // C-array
  template <typename T, std::size_t n>
  typename std::enable_if<!is_compact<T, n>::value, void>::type operator()(
      const char *key,
      TArray<T, n> &val) {
    add_line(key, "[");
    indent++;
    for (std::size_t i = 0; i < n; i++) {
      this->operator()(("[" + std::to_string(i) + "]").c_str(), val[i]);
    }
    indent--;
    add_line("]");
  }

  // Elementary data types
  template <typename T>
  typename std::enable_if<!has_io<T>::value, void>::type operator()(
      const char *key,
      T &&val) {
    static_assert(!has_io<T>::value, "");
    std::stringstream ss;
    ss << std::boolalpha << val;
    add_line(key, ss.str());
  }

  template <typename T>
  typename std::enable_if<has_io<T>::value, void>::type operator()(
      const char *key,
      T &&val) {
    add_line(key, "{");
    indent++;
    val.io(*this);
    indent--;
    add_line("}");
  }

  template <typename T>
  void operator()(const char *key, std::vector<T> &val) {
    add_line(key, "[");
    indent++;
    for (std::size_t i = 0; i < val.size(); i++) {
      this->operator()(("[" + std::to_string(i) + "]").c_str(),
                       std::forward<T>(val[i]));
    }
    indent--;
    add_line("]");
  }

  template <typename T, typename G>
  void operator()(const char *key, std::pair<T, G> &val) {
    add_line(key, "(");
    indent++;
    this->operator()("[0]", val.first);
    this->operator()("[1]", val.second);
    indent--;
    add_line(")");
  }

  template <typename T, typename G>
  void operator()(const char *key, std::map<T, G> &val) {
    add_line(key, "{");
    indent++;
    for (auto iter : val) {
      T first = iter.first;
      this->operator()(nullptr, first);
      this->operator()(nullptr, iter.second);
    }
    indent--;
    add_line("}");
  }

  template <typename T, typename... Args>
  void operator()(const char *key_, T &&t, Args &&... rest) {
    std::string key(key_);
    size_t pos = key.find(",");
    std::string first_name = key.substr(0, pos);
    std::string rest_names =
        key.substr(pos + 2, int(key.size()) - (int)pos - 2);
    this->operator()(first_name.c_str(), std::forward<T>(t));
    this->operator()(rest_names.c_str(), std::forward<Args>(rest)...);
  }

  template <typename T>
  void operator()(Item<T> &item) {
    this->operator()(item.key.c_str(), item.value);
  }
};

TC_NAMESPACE_END
