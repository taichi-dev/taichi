#pragma once

#include <functional>
#include <string>

namespace taichi {
namespace lang {

class FunctionKey {
 public:
  std::string func_name;
  int func_id;
  int instance_id;

  FunctionKey(const std::string &func_name, int func_id, int instance_id);

  bool operator==(const FunctionKey &other_key) const;

  [[nodiscard]] std::string get_full_name() const;
};

}  // namespace lang
}  // namespace taichi

namespace std {
template <>
struct hash<taichi::lang::FunctionKey> {
  std::size_t operator()(const taichi::lang::FunctionKey &key) const noexcept {
    return key.func_id ^ (key.instance_id << 16);
  }
};
}  // namespace std
