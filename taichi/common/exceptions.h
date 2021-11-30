#pragma once

namespace taichi {
namespace lang {

class IRModified {};

class TaichiTypeError : public std::exception {
  std::string msg_;

 public:
  TaichiTypeError(const std::string msg) : msg_(msg) {
  }
  const char *what() const throw() override {
    return msg_.c_str();
  }
};

}  // namespace lang
}  // namespace taichi
