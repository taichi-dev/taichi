/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <array>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifdef TI_INCLUDED
TI_NAMESPACE_BEGIN
#else
#define TI_NAMESPACE_BEGIN
#define TI_NAMESPACE_END
#define TI_TRACE
#define TI_CRITICAL
#define TI_ASSERT assert
#endif

template <typename T>
std::unique_ptr<T> create_instance_unique(const std::string &alias);

////////////////////////////////////////////////////////////////////////////////
//                   A Minimalist Serializer for Taichi                       //
//                           (Requires C++17)                                 //
////////////////////////////////////////////////////////////////////////////////

// TODO: Consider using third-party serialization libraries
// * https://github.com/USCiLab/cereal
class Unit;

namespace type {

template <typename T>
using remove_cvref =
    typename std::remove_cv<typename std::remove_reference<T>::type>;

template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

template <typename T>
using is_unit = typename std::is_base_of<Unit, remove_cvref_t<T>>;

template <typename T>
using is_unit_t = typename is_unit<T>::type;

}  // namespace type
class TextSerializer;
namespace detail {

template <size_t N>
constexpr size_t count_delim(const char (&str)[N], char delim) {
  size_t count = 1;
  for (const char &ch : str) {
    if (ch == delim) {
      ++count;
    }
  }
  return count;
}

template <size_t DelimN>
struct StrDelimSplitter {
  template <size_t StrsN>
  static constexpr std::array<std::string_view, DelimN> make(
      const char (&str)[StrsN],
      char delim) {
    std::array<std::string_view, DelimN> res;
    const char *head = &(str[0]);
    size_t si = 0;
    size_t cur_head_i = 0;
    size_t ri = 0;
    while (si < StrsN) {
      if (str[si] != delim) {
        si += 1;
      } else {
        res[ri] = {head, (si - cur_head_i)};
        ++ri;
        si += 2;  // skip ", "
        cur_head_i = si;
        head = &(str[cur_head_i]);
      }
    }
    // `StrsN - 1` because the last char is '\0'.
    res[ri] = {head, (StrsN - 1 - cur_head_i)};
    return res;
  }
};

template <typename SER, size_t N, typename T>
void serialize_kv_impl(SER &ser,
                       const std::array<std::string_view, N> &keys,
                       T &&val) {
  std::string key{keys[N - 1]};
  ser(key.c_str(), val);
}

template <typename SER, size_t N, typename T, typename... Args>
typename std::enable_if<!std::is_same<SER, TextSerializer>::value, void>::type
serialize_kv_impl(SER &ser,
                  const std::array<std::string_view, N> &keys,
                  T &&head,
                  Args &&...rest) {
  constexpr auto i = (N - 1 - sizeof...(Args));
  std::string key{keys[i]};
  ser(key.c_str(), head);
  serialize_kv_impl(ser, keys, rest...);
}

// Specialize for TextSerializer since we need to append comma in the end for
// non-last object.
template <typename SER, size_t N, typename T, typename... Args>
typename std::enable_if<std::is_same<SER, TextSerializer>::value, void>::type
serialize_kv_impl(SER &ser,
                  const std::array<std::string_view, N> &keys,
                  T &&head,
                  Args &&...rest) {
  constexpr auto i = (N - 1 - sizeof...(Args));
  std::string key{keys[i]};
  ser(key.c_str(), head, true);
  serialize_kv_impl(ser, keys, rest...);
}

}  // namespace detail

#define TI_IO_DECL      \
  template <typename S> \
  void io(S &serializer) const

#define TI_IO_DEF(...)           \
  template <typename S>          \
  void io(S &serializer) const { \
    TI_IO(__VA_ARGS__);          \
  }

// This macro serializes each field with its name by doing the following:
// 1. Stringifies __VA_ARGS__, then split the stringified result by ',' at
// compile time.
// 2. Invoke serializer::operator("arg", arg) for each arg in __VA_ARGS__. This
// is implemented inside detail::serialize_kv_impl.
#define TI_IO(...)                                                     \
  do {                                                                 \
    constexpr size_t kDelimN = detail::count_delim(#__VA_ARGS__, ','); \
    constexpr auto kSplitStrs =                                        \
        detail::StrDelimSplitter<kDelimN>::make(#__VA_ARGS__, ',');    \
    detail::serialize_kv_impl(serializer, kSplitStrs, __VA_ARGS__);    \
  } while (0)

#define TI_SERIALIZER_IS(T)                                                 \
  (std::is_same<typename std::remove_reference<decltype(serializer)>::type, \
                T>())

#if !defined(TI_ARCH_x86)
static_assert(
    sizeof(std::size_t) == sizeof(uint64_t),
    "sizeof(std::size_t) should be 8. Try compiling with 64bit mode.");
#endif
template <typename T, typename S>
struct IO {
  using implemented = std::false_type;
};

class Serializer {
 public:
  template <typename T, std::size_t n>
  using TArray = T[n];
  template <typename T, std::size_t n>
  using StdTArray = std::array<T, n>;

  std::unordered_map<std::size_t, void *> assets;

  template <typename T, typename T_ = typename type::remove_cvref_t<T>>
  static T_ &get_writable(T &&t) {
    return *const_cast<T_ *>(&t);
  }

  template <typename T>
  struct has_io {
    template <typename T_>
    static constexpr auto helper(T_ *) -> std::is_same<
        decltype((std::declval<T_>().io(std::declval<Serializer &>()))),
        void>;

    template <typename>
    static constexpr auto helper(...) -> std::false_type;

   public:
    using T__ = typename type::remove_cvref_t<T>;
    using type = decltype(helper<T__>(nullptr));
    static constexpr bool value = type::value;
  };

  template <typename T>
  struct has_free_io {
    template <typename T_>
    static constexpr auto helper(T_ *) ->
        typename IO<T_, Serializer>::implemented;

    template <typename>
    static constexpr auto helper(...) -> std::false_type;

   public:
    using T__ = typename type::remove_cvref_t<T>;
    using type = decltype(helper<T__>(nullptr));
    static constexpr bool value = type::value;
  };
};

inline std::vector<uint8_t> read_data_from_file(const std::string &fn) {
  std::vector<uint8_t> data;
  std::FILE *f = fopen(fn.c_str(), "rb");
  if (f == nullptr) {
    TI_DEBUG("Cannot open file: {}", fn);
    return std::vector<uint8_t>();
  }
  if (ends_with(fn, ".zip")) {
    std::fclose(f);
    // Read zip file, e.g. particles.tcb.zip
    return zip::read(fn);
  } else {
    // Read uncompressed file, e.g. particles.tcb
    assert(f != nullptr);
    std::size_t length = 0;
    while (true) {
      size_t limit = 1 << 8;
      data.resize(data.size() + limit);
      void *ptr = reinterpret_cast<void *>(&data[length]);
      size_t length_tmp = fread(ptr, sizeof(uint8_t), limit, f);
      length += length_tmp;
      if (length_tmp < limit) {
        break;
      }
    }
    std::fclose(f);
    data.resize(length);
    return data;
  }
}

inline void write_data_to_file(const std::string &fn,
                               uint8_t *data,
                               std::size_t size) {
  std::FILE *f = fopen(fn.c_str(), "wb");
  if (f == nullptr) {
    TI_ERROR("Cannot open file [{}] for writing. (Does the directory exist?)",
             fn);
    assert(f != nullptr);
  }
  if (ends_with(fn, ".tcb.zip")) {
    std::fclose(f);
    zip::write(fn, data, size);
  } else if (ends_with(fn, ".tcb")) {
    fwrite(data, sizeof(uint8_t), size, f);
    std::fclose(f);
  } else {
    TI_ERROR("File must end with .tcb or .tcb.zip. [Filename = {}]", fn);
  }
}

template <bool writing>
class BinarySerializer : public Serializer {
 private:
  template <typename T>
  inline static constexpr bool is_elementary_type_v =
      !has_io<T>::value && !std::is_pointer<T>::value && !std::is_enum_v<T> &&
      std::is_pod_v<T>;

 public:
  std::vector<uint8_t> data;
  uint8_t *c_data;

  std::size_t head;
  std::size_t preserved;

  using Base = Serializer;
  using Base::assets;

  template <bool writing_ = writing>
  typename std::enable_if<!writing_, bool>::type initialize(
      const std::string &fn) {
    data = read_data_from_file(fn);
    if (data.size() == 0) {
      return false;
    }
    c_data = reinterpret_cast<uint8_t *>(&data[0]);
    head = sizeof(std::size_t);
    return true;
  }

  void write_to_file(const std::string &fn) {
    void *ptr = c_data;
    if (!ptr) {
      assert(!data.empty());
      ptr = &data[0];
    }
    write_data_to_file(fn, reinterpret_cast<uint8_t *>(ptr), head);
  }

  template <bool writing_ = writing>
  typename std::enable_if<writing_, bool>::type initialize(
      std::size_t preserved_ = std::size_t(0),
      void *c_data = nullptr) {
    std::size_t n = 0;
    head = 0;
    if (preserved_ != 0) {
      TI_TRACE("preserved = {}", preserved_);
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
    return true;
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

  template <typename T>
  void operator()(const char *, const T &val) {
    this->process(val);
  }

  template <typename T>
  void operator()(const T &val) {
    this->process(val);
  }

 private:
  // std::string
  void process(const std::string &val_) {
    auto &val = get_writable(val_);
    if (writing) {
      std::vector<char> val_vector(val.begin(), val.end());
      this->process(val_vector);
    } else {
      std::vector<char> val_vector;
      this->process(val_vector);
      val = std::string(val_vector.begin(), val_vector.end());
    }
  }

  // C-array
  template <typename T, std::size_t n>
  void process(const TArray<T, n> &val) {
    if (writing) {
      for (std::size_t i = 0; i < n; i++) {
        this->process(val[i]);
      }
    } else {
      // TODO: why do I have to let it write to tmp, otherwise I get Sig Fault?
      // Take care of std::vector<bool> ...
      using Traw = typename type::remove_cvref_t<T>;
      std::vector<
          std::conditional_t<std::is_same<Traw, bool>::value, uint8, Traw>>
          tmp(n);
      for (std::size_t i = 0; i < n; i++) {
        this->process(tmp[i]);
      }
      std::memcpy(const_cast<typename std::remove_cv<T>::type *>(val), &tmp[0],
                  sizeof(tmp[0]) * tmp.size());
    }
  }

  // Elementary data types
  template <typename T>
  typename std::enable_if_t<is_elementary_type_v<T>, void> process(
      const T &val) {
    static_assert(!std::is_reference<T>::value, "T cannot be reference");
    static_assert(!std::is_const<T>::value, "T cannot be const");
    static_assert(!std::is_volatile<T>::value, "T cannot be volatile");
    static_assert(!std::is_pointer<T>::value, "T cannot be pointer");
    if (writing) {
      std::size_t new_size = head + sizeof(T);
      if (c_data) {
        if (new_size > preserved) {
          TI_CRITICAL("Preserved Buffer (size {}) Overflow.", preserved);
        }
        //*reinterpret_cast<typename type::remove_cvref_t<T> *>(&c_data[head]) =
        //    val;
        std::memcpy(&c_data[head], &val, sizeof(T));
      } else {
        data.resize(new_size);
        //*reinterpret_cast<typename type::remove_cvref_t<T> *>(&data[head]) =
        //    val;
        std::memcpy(&data[head], &val, sizeof(T));
      }
    } else {
      // get_writable(val) =
      //    *reinterpret_cast<typename std::remove_reference<T>::type *>(
      //        &c_data[head]);
      std::memcpy(&get_writable(val), &c_data[head], sizeof(T));
    }
    head += sizeof(T);
  }

  template <typename T>
  std::enable_if_t<has_io<T>::value, void> process(const T &val) {
    val.io(*this);
  }

  // Unique Pointers to non-taichi-unit Types
  template <typename T>
  typename std::enable_if<!type::is_unit<T>::value, void>::type process(
      const std::unique_ptr<T> &val_) {
    auto &val = get_writable(val_);
    if (writing) {
      this->process(ptr_to_int(val.get()));
      if (val.get() != nullptr) {
        this->process(*val);
        // Just for checking future raw pointers
        assets.insert(std::make_pair(ptr_to_int(val.get()), val.get()));
      }
    } else {
      std::size_t original_addr;
      this->process(original_addr);
      if (original_addr != 0) {
        val = std::make_unique<T>();
        assets.insert(std::make_pair(original_addr, val.get()));
        this->process(*val);
      }
    }
  }

  template <typename T>
  std::size_t ptr_to_int(T *t) {
    return reinterpret_cast<std::size_t>(t);
  }

  // Unique Pointers to taichi-unit Types
  template <typename T>
  typename std::enable_if<type::is_unit<T>::value, void>::type process(
      const std::unique_ptr<T> &val_) {
    auto &val = get_writable(val_);
    if (writing) {
      this->process(val->get_name());
      this->process(ptr_to_int(val.get()));
      if (val.get() != nullptr) {
        val->binary_io(nullptr, *this);
        // Just for checking future raw pointers
        assets.insert(std::make_pair(ptr_to_int(val.get()), val.get()));
      }
    } else {
      std::string name;
      std::size_t original_addr;
      this->process(name);
      this->process(original_addr);
      if (original_addr != 0) {
        val = create_instance_unique<T>(name);
        assets.insert(std::make_pair(original_addr, val.get()));
        val->binary_io(nullptr, *this);
      }
    }
  }

  // Raw pointers (no ownership)
  template <typename T>
  typename std::enable_if<std::is_pointer<T>::value, void>::type process(
      const T &val_) {
    auto &val = get_writable(val_);
    if (writing) {
      this->process(ptr_to_int(val));
      if (val != nullptr) {
        TI_ASSERT_INFO(assets.find(ptr_to_int(val)) != assets.end(),
                       "Cannot find the address with a smart pointer pointing "
                       "to. Make sure the smart pointer is serialized before "
                       "the raw pointer.");
      }
    } else {
      std::size_t val_ptr = 0;
      this->process(val_ptr);
      if (val_ptr != 0) {
        TI_ASSERT(assets.find(val_ptr) != assets.end());
        val = reinterpret_cast<typename std::remove_pointer<T>::type *>(
            assets[val_ptr]);
      }
    }
  }

  // enum class
  template <typename T>
  typename std::enable_if<std::is_enum_v<T>, void>::type process(const T &val) {
    using UT = std::underlying_type_t<T>;
    // https://stackoverflow.com/a/62688905/12003165
    if constexpr (writing) {
      this->process(static_cast<UT>(val));
    } else {
      auto &wval = get_writable(val);
      UT &underlying_wval = reinterpret_cast<UT &>(wval);
      this->process(underlying_wval);
    }
  }

  // std::vector
  template <typename T>
  void process(const std::vector<T> &val_) {
    auto &val = get_writable(val_);
    if (writing) {
      this->process(val.size());
    } else {
      std::size_t n = 0;
      this->process(n);
      val.resize(n);
    }
    for (std::size_t i = 0; i < val.size(); i++) {
      this->process(val[i]);
    }
  }

  // std::pair
  template <typename T, typename G>
  void process(const std::pair<T, G> &val) {
    this->process(val.first);
    this->process(val.second);
  }

  // std::map
  template <typename K, typename V>
  void process(const std::map<K, V> &val) {
    handle_associative_container(val);
  }

  // std::unordered_map
  template <typename K, typename V>
  void process(const std::unordered_map<K, V> &val) {
    handle_associative_container(val);
  }

  // std::optional
  template <typename T>
  void process(const std::optional<T> &val) {
    if constexpr (writing) {
      this->process(val.has_value());
      if (val.has_value()) {
        this->process(val.value());
      }
    } else {
      bool has_value{false};
      this->process(has_value);
      auto &wval = get_writable(val);
      if (!has_value) {
        wval.reset();
      } else {
        T new_val;
        this->process(new_val);
        wval = std::move(new_val);
      }
    }
  }

  template <typename M>
  void handle_associative_container(const M &val) {
    if constexpr (writing) {
      this->process(val.size());
      for (auto &iter : val) {
        auto first = iter.first;
        this->process(first);
        this->process(iter.second);
      }
    } else {
      auto &wval = get_writable(val);
      wval.clear();
      std::size_t n = 0;
      this->process(n);
      for (std::size_t i = 0; i < n; i++) {
        typename M::value_type record;
        this->process(record);
        wval.insert(std::move(record));
      }
    }
  }
};

using BinaryOutputSerializer = BinarySerializer<true>;
using BinaryInputSerializer = BinarySerializer<false>;

// Serialize to JSON format
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
  int indent_;
  static constexpr int indent_width = 2;
  bool first_line_;

  template <typename T>
  inline static constexpr bool is_elementary_type_v =
      !has_io<T>::value && !has_free_io<T>::value && !std::is_enum_v<T> &&
      std::is_pod_v<T>;

 public:
  TextSerializer() {
    indent_ = 0;
    first_line_ = false;
  }

  template <typename T>
  static std::string serialize(const char *key, const T &t) {
    TextSerializer ser;
    ser(key, t);
    return ser.data;
  }

  template <typename T>
  void operator()(const char *key, const T &t, bool append_comma = false) {
    add_key(key);
    process(t);
    if (append_comma) {
      add_raw(",");
    }
  }

  // Entry to make an AOT json file
  template <typename T>
  void serialize_to_json(const char *key, const T &t) {
    add_raw("{");
    (*this)(key, t);
    add_raw("}");
  }

 private:
  void process(const std::string &val) {
    add_raw("\"" + val + "\"");
  }

  template <typename T, std::size_t n>
  using is_compact =
      typename std::integral_constant<bool,
                                      std::is_arithmetic<T>::value && (n < 7)>;

  // C-array
  template <typename T, std::size_t n>
  std::enable_if_t<is_compact<T, n>::value, void> process(
      const TArray<T, n> &val) {
    std::stringstream ss;
    ss << "{";
    for (std::size_t i = 0; i < n; i++) {
      ss << val[i];
      if (i != n - 1) {
        ss << ", ";
      }
    }
    ss << "}";
    add_raw(ss.str());
  }

  // C-array
  template <typename T, std::size_t n>
  std::enable_if_t<!is_compact<T, n>::value, void> process(
      const TArray<T, n> &val) {
    add_raw("{");
    indent_++;
    for (std::size_t i = 0; i < n; i++) {
      add_key(std::to_string(i).c_str());
      process(val[i]);
      if (i != n - 1) {
        add_raw(",");
      }
    }
    indent_--;
    add_raw("}");
  }

  // std::array
  template <typename T, std::size_t n>
  std::enable_if_t<is_compact<T, n>::value, void> process(
      const StdTArray<T, n> &val) {
    std::stringstream ss;
    ss << "{";
    for (std::size_t i = 0; i < n; i++) {
      ss << val[i];
      if (i != n - 1) {
        ss << ", ";
      }
    }
    ss << "}";
    add_raw(ss.str());
  }

  // std::array
  template <typename T, std::size_t n>
  std::enable_if_t<!is_compact<T, n>::value, void> process(
      const StdTArray<T, n> &val) {
    add_raw("{");
    indent_++;
    for (std::size_t i = 0; i < n; i++) {
      add_key(std::to_string(i).c_str());
      process(val[i]);
      if (i != n - 1) {
        add_raw(",");
      }
    }
    indent_--;
    add_raw("}");
  }

  // Elementary data types
  template <typename T>
  std::enable_if_t<is_elementary_type_v<T>, void> process(const T &val) {
    static_assert(!has_io<T>::value, "");
    std::stringstream ss;
    ss << std::boolalpha << val;
    add_raw(ss.str());
  }

  template <typename T>
  std::enable_if_t<has_io<T>::value, void> process(const T &val) {
    add_raw("{");
    indent_++;
    val.io(*this);
    indent_--;
    add_raw("}");
  }

  template <typename T>
  std::enable_if_t<has_free_io<T>::value, void> process(const T &val) {
    add_raw("{");
    indent_++;
    IO<typename type::remove_cvref_t<T>, decltype(*this)>()(*this, val);
    indent_--;
    add_raw("}");
  }

  template <typename T>
  std::enable_if_t<std::is_enum_v<T>, void> process(const T &val) {
    using UT = std::underlying_type_t<T>;
    process(static_cast<UT>(val));
  }

  template <typename T>
  void process(const std::vector<T> &val) {
    add_raw("[");
    indent_++;
    for (std::size_t i = 0; i < val.size(); i++) {
      process(val[i]);
      if (i < val.size() - 1) {
        add_raw(",");
      }
    }
    indent_--;
    add_raw("]");
  }

  template <typename T, typename G>
  void process(const std::pair<T, G> &val) {
    add_raw("[");
    indent_++;
    process("first", val.first);
    add_raw(", ");
    process("second", val.second);
    indent_--;
    add_raw("]");
  }

  // std::map
  template <typename K, typename V>
  void process(const std::map<K, V> &val) {
    handle_associative_container(val);
  }

  // std::unordered_map
  template <typename K, typename V>
  void process(const std::unordered_map<K, V> &val) {
    handle_associative_container(val);
  }

  // std::optional
  template <typename T>
  void process(const std::optional<T> &val) {
    add_raw("{");
    indent_++;
    add_key("has_value");
    process(val.has_value());
    if (val.has_value()) {
      add_raw(",");
      add_key("value");
      process(val.value());
    }
    indent_--;
    add_raw("}");
  }

  template <typename M>
  void handle_associative_container(const M &val) {
    add_raw("{");
    indent_++;
    for (auto iter = val.begin(); iter != val.end(); iter++) {
      auto first = iter->first;
      bool is_string = typeid(first) == typeid(std::string);
      // Non-string keys must be wrapped by quotes.
      if (!is_string) {
        add_raw("\"");
      }
      process(first);
      if (!is_string) {
        add_raw("\"");
      }
      add_raw(": ");
      process(iter->second);
      if (std::next(iter) != val.end()) {
        add_raw(",");
      }
    }
    indent_--;
    add_raw("}");
  }

  void add_raw(const std::string &str) {
    data += str;
  }

  void add_key(const std::string &key) {
    if (first_line_) {
      first_line_ = false;
    } else {
      data += "\n";
    }
    data += std::string(indent_width * indent_, ' ') + "\"" + key + "\"";

    add_raw(": ");
  }
};

template <typename T>
typename std::enable_if<Serializer::has_io<T>::value, std::ostream &>::type
operator<<(std::ostream &os, const T &t) {
  os << TextSerializer::serialize("value", t);
  return os;
}

// Returns true if deserialization succeeded.
template <typename T>
bool read_from_binary_file(T &t, const std::string &file_name) {
  BinaryInputSerializer reader;
  if (!reader.initialize(file_name)) {
    return false;
  }
  reader(t);
  reader.finalize();
  return true;
}

template <typename T>
void write_to_binary_file(const T &t, const std::string &file_name) {
  BinaryOutputSerializer writer;
  writer.initialize();
  writer(t);
  writer.finalize();
  writer.write_to_file(file_name);
}

// Compile-Time Tests
static_assert(std::is_same<decltype(Serializer::get_writable(
                               std::declval<const std::vector<int> &>())),
                           std::vector<int> &>(),
              "");

static_assert(
    std::is_same<
        decltype(Serializer::get_writable(
            std::declval<const std::vector<std::unique_ptr<int>> &>())),
        std::vector<std::unique_ptr<int>> &>(),
    "");

TI_NAMESPACE_END
