#pragma once

namespace taichi::lang {

class IRModified {};

class TaichiExceptionImpl : public std::exception {
  friend struct ErrorEmitter;

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

class TaichiTypeError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;
};

class TaichiSyntaxError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;
};

class TaichiIndexError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;
};

class TaichiRuntimeError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;
};

class TaichiAssertionError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;
};

}  // namespace taichi::lang
