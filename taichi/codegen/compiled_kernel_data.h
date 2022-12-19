#pragma once

#include <any>
#include <memory>
#include <optional>

namespace taichi::lang {

class CompiledKernelData {
 public:
  enum class Err { kNoError, kFailed, kMaxErr };

  CompiledKernelData() = default;
  CompiledKernelData(const CompiledKernelData &) = delete;
  CompiledKernelData &operator=(const CompiledKernelData &) = delete;
  virtual ~CompiledKernelData() = default;

  virtual std::size_t size() const = 0;

  virtual Err load(std::istream &is) = 0;
  virtual Err dump(std::ostream &os) const = 0;
  virtual std::unique_ptr<CompiledKernelData> clone() const = 0;

  virtual Err debug_print(std::ostream &os) const {
    return dump(os);
  }

  virtual Err check() const {
    return Err::kNoError;
  }

  // Refactor2023:FIXME: Temp solution. Remove it after making Runtime stateless
  void set_handle(std::any handle) {
    handle_ = std::move(handle);
  }

  // Refactor2023:FIXME: Temp solution. Remove it after making Runtime stateless
  const std::optional<std::any> &get_handle() const {
    return handle_;
  }

 private:
  // Refactor2023:FIXME: Temp solution. Remove it after making Runtime stateless
  std::optional<std::any> handle_;
};

}  // namespace taichi::lang
