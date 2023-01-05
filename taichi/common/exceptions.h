#pragma once

namespace taichi::lang {

class IRModified {};

class TaichiExceptionImpl : public std::exception {
  std::string msg_;

 public:
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
