#pragma once

namespace taichi::lang {

class IRModified {};

class TaichiExceptionImpl : public std::exception {
  friend struct ErrorEmitter;

 private:
  virtual void emit() = 0;

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

class TaichiTypeError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;

  [[noreturn]] void emit() override {
    throw *this;
  }
};

class TaichiSyntaxError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;

  [[noreturn]] void emit() override {
    throw *this;
  }
};

class TaichiIndexError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;

  [[noreturn]] void emit() override {
    throw *this;
  }
};

class TaichiRuntimeError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;

  [[noreturn]] void emit() override {
    throw *this;
  }
};

class TaichiAssertionError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;

  [[noreturn]] void emit() override {
    throw *this;
  }
};

class TaichiIrError : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;

  [[noreturn]] void emit() override {
    throw *this;
  }
};

class TaichiCastWarning : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;

  void emit() noexcept override {
    taichi::Logger::get_instance().warn("TaichiCastWarning\n" + msg_);
  }
};

class TaichiTypeWarning : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;

  void emit() noexcept override {
    taichi::Logger::get_instance().warn("TaichiTypeWarning\n" + msg_);
  }
};

class TaichiIrWarning : public TaichiExceptionImpl {
  using TaichiExceptionImpl::TaichiExceptionImpl;

  void emit() noexcept override {
    taichi::Logger::get_instance().warn("TaichiIrWarning\n" + msg_);
  }
};

}  // namespace taichi::lang
