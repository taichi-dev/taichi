#pragma once

#include "taichi/lang_util.h"

namespace taichi {
namespace lang {

class Program;
class IRNode;
class FrontendContext;

class Callable {
 public:
  Program *program;
  std::unique_ptr<IRNode> ir;
  std::unique_ptr<FrontendContext> context;

  struct Arg {
    PrimitiveTypeID ptid;
    bool is_external_array;
    std::size_t size;

    explicit Arg(const PrimitiveTypeID &ptid = PrimitiveTypeID::unknown,
                 bool is_external_array = false,
                 std::size_t size = 0)
        : ptid(ptid), is_external_array(is_external_array), size(size) {
    }
  };

  struct Ret {
    PrimitiveTypeID ptid;

    explicit Ret(const PrimitiveTypeID &ptid = PrimitiveTypeID::unknown)
        : ptid(ptid) {
    }
  };

  std::vector<Arg> args;
  std::vector<Ret> rets;

  virtual ~Callable() = default;

  int insert_arg(const PrimitiveTypeID &ptid, bool is_external_array);

  int insert_ret(const PrimitiveTypeID &ptid);

  [[nodiscard]] virtual std::string get_name() const = 0;

  class CurrentCallableGuard {
    Callable *old_callable;
    Program *program;

   public:
    CurrentCallableGuard(Program *program, Callable *callable);

    ~CurrentCallableGuard();
  };
};

}  // namespace lang
}  // namespace taichi
