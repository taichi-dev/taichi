#pragma once

namespace taichi {
namespace lang {

class IRModified {};

class TaichiExceptionImpl : public std::exception {
  std::string msg_;

 public:
  TaichiExceptionImpl(const std::string msg) : msg_(msg) {
  }
  const char *what() const throw() override {
    return msg_.c_str();
  }
};

class TaichiTypeError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;
};

class TaichiSyntaxError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;
};

class TaichiRuntimeError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;
};

}  // namespace lang
}  // namespace taichi
