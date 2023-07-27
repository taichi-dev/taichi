#pragma once

#include <exception>
#include <string>
#include <string_view>
#include "taichi/common/logging.h"

namespace taichi::lang {

class IRModified {};

struct Location {
  int line_number = 0;
  std::string var_name = "";
};

struct DebugInfo {
  Location src_loc;
  std::string tb;

  explicit DebugInfo() = default;

  explicit DebugInfo(std::string tb_) : tb(tb_) {
  }

  explicit DebugInfo(const char *tb_) : tb(tb_) {
  }

  std::string const &get_tb() const {
    return tb;
  }

  void set_tb(std::string const &tb) {
    this->tb = tb;
  }
};

class TaichiExceptionImpl : public std::exception {
  friend struct ErrorEmitter;

 protected:
  std::string msg_;

 public:
  // Add default constructor to allow passing Exception to ErrorEmitter
  // TODO: remove this and find a better way
  explicit TaichiExceptionImpl() = default;
  explicit TaichiExceptionImpl(const std::string msg) : msg_(msg) {
  }
  const char *what() const noexcept override {
    return msg_.c_str();
  }
};

class TaichiError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;
};

class TaichiWarning : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;

 protected:
  static constexpr std::string_view name_;

 public:
  void emit() {
    taichi::Logger::get_instance().warn(std::string(name_) + "\n" + msg_);
  }
};

class TaichiTypeError : public TaichiError {
  using TaichiError::TaichiError;
};

class TaichiSyntaxError : public TaichiError {
  using TaichiError::TaichiError;
};

class TaichiIndexError : public TaichiError {
  using TaichiError::TaichiError;
};

class TaichiRuntimeError : public TaichiError {
  using TaichiError::TaichiError;
};

class TaichiAssertionError : public TaichiError {
  using TaichiError::TaichiError;
};

class TaichiIrError : public TaichiError {
  using TaichiError::TaichiError;
};

class TaichiCastWarning : public TaichiWarning {
  using TaichiWarning::TaichiWarning;
  static constexpr std::string_view name_ = "TaichiCastWarning";
};

class TaichiTypeWarning : public TaichiWarning {
  using TaichiWarning::TaichiWarning;
  static constexpr std::string_view name_ = "TaichiTypeWarning";
};

class TaichiIrWarning : public TaichiWarning {
  using TaichiWarning::TaichiWarning;
  static constexpr std::string_view name_ = "TaichiIrWarning";
};

class TaichiIndexWarning : public TaichiWarning {
  using TaichiWarning::TaichiWarning;
  static constexpr std::string_view name_ = "TaichiIndexWarning";
};

class TaichiRuntimeWarning : public TaichiWarning {
  using TaichiWarning::TaichiWarning;
  static constexpr std::string_view name_ = "TaichiRuntimeWarning";
};

struct ErrorEmitter {
  ErrorEmitter() = delete;
  ErrorEmitter(ErrorEmitter &) = delete;
  ErrorEmitter(ErrorEmitter &&) = delete;

  // Emit an error on stmt with error message
  template <typename E,
            typename = std::enable_if_t<
                std::is_base_of_v<TaichiExceptionImpl, std::decay_t<E>>>,
            // The expected type for T is `Stmt`, `Expression`, or `DebugInfo`.
            // These types have a member function named get_tb() that returns
            // trace back information as a `std::string`.
            typename T,
            typename = std::enable_if_t<std::is_same_v<
                std::decay_t<decltype(std::declval<T>()->get_tb())>,
                std::string>>>
  ErrorEmitter(E &&error, T p_dbg_info, std::string &&error_msg) {
    if constexpr ((std::is_same_v<std::decay_t<T>, DebugInfo *> ||
                   std::is_same_v<std::decay_t<T>, const DebugInfo *>)&&std::
                      is_base_of_v<TaichiError, std::decay_t<E>>) {
      // Indicates a failed C++ API call from Python side, we should not print
      // tb here
      error.msg_ = error_msg;
    } else {
      error.msg_ = p_dbg_info->get_tb() + error_msg;
    }

    if constexpr (std::is_base_of_v<TaichiWarning, std::decay_t<E>>) {
      error.emit();
    } else if constexpr (std::is_base_of_v<TaichiError, std::decay_t<E>>) {
      throw std::move(error);
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }
};

}  // namespace taichi::lang
