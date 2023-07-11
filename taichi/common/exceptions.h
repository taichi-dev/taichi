#pragma once

#include <exception>
#include <string>
#include <string_view>
namespace taichi::lang {

class IRModified {};
struct DebugInfo;

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
            typename T,
            typename = std::enable_if_t<
                std::is_same_v<std::decay_t<decltype(std::declval<T>()->tb)>,
                               std::string>>>
  ErrorEmitter(E &&error, T p_stmt, std::string &&error_msg) {
    if constexpr ((std::is_same_v<std::decay_t<T>, DebugInfo *> ||
                   std::is_same_v<std::decay_t<T>, const DebugInfo *>)&&std::
                      is_base_of_v<TaichiError, std::decay_t<E>>) {
      // Indicates a failed C++ API call from Python side, we should not print
      // tb here
      error.msg_ = error_msg;
    } else {
      error.msg_ = p_stmt->tb + error_msg;
    }

    if constexpr (std::is_base_of_v<TaichiWarning, std::decay_t<E>>) {
      error.emit();
    } else if constexpr (std::is_base_of_v<TaichiError, std::decay_t<E>>) {
      throw error;
    } else {
      TI_STOP;
    }
  }

  // Emit an error when expression is false
  template <typename E, typename T>
  ErrorEmitter(bool expression, E &&error, T p_stmt, std::string &&error_msg) {
    if (!expression) {
      ErrorEmitter(std::move(error), p_stmt, std::move(error_msg));
    }
  }
};

}  // namespace taichi::lang
