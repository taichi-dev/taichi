#pragma once

#include <functional>
#include <string>

namespace taichi {
namespace lang {

/**
 * A unique key of a function.
 * |func_id| uniquely corresponds to a function template (@ti.func).
 * |instance_id| is for the template instantiations, i.e., a @ti.func
 * instantiated multiple times with different ti.template() parameters.
 * |func_name| is mostly for debugging/visualizing purpose, and doesn't need
 * to participate in the hash computation.
 */
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
