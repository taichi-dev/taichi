#pragma once

#include "taichi/lang_util.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

class Program;

class Kernel {
 public:
  std::unique_ptr<IRNode> ir;
  Program &program;
  FunctionType compiled;
  std::string name;
  std::vector<SNode *> no_activate;
  Arch arch;
  bool lowered;  // lower inital AST all the way down to a bunch of
                 // OffloadedStmt for async execution

  struct Arg {
    DataType dt;
    bool is_nparray;
    std::size_t size;

    Arg(DataType dt = DataType::unknown,
        bool is_nparray = false,
        std::size_t size = 0)
        : dt(dt), is_nparray(is_nparray), size(size) {
    }
  };

  struct Ret {
    DataType dt;

    explicit Ret(DataType dt = DataType::unknown) : dt(dt) {
    }
  };

  std::vector<Arg> args;
  std::vector<Ret> rets;
  bool is_accessor;
  bool is_evaluator;
  bool grad;

  // TODO: Give "Context" a more specific name.
  class LaunchContextBuilder {
   public:
    LaunchContextBuilder(Kernel *kernel, Context *ctx)
        : kernel_(kernel), ctx_(ctx) {
    }

    void set_arg_float(int i, float64 d);

    void set_arg_int(int i, int64 d);

    void set_extra_arg_int(int i, int j, int32 d);

    void set_arg_nparray(int i, uint64 ptr, uint64 size);

    // Sets the i-th arg in the context to the bits stored in |d|. This ignores
    // the underlying kernel's i-th arg type.
    void set_arg_raw(int i, uint64 d);

    Context &get_context();

   private:
    Kernel *const kernel_;
    // TODO: Right now |ctx_| is borrowed from other places: either the
    // program's context, or the one in the CUDA launch function. In the future,
    // this could *own* a Context (possibly through a std::unique_ptr, since we
    // don't always need to own the Context.)
    Context *const ctx_;
  };

  Kernel(Program &program,
         std::function<void()> func,
         std::string name = "",
         bool grad = false);

  void compile();

  void lower(bool to_executable = true);

  void operator()(LaunchContextBuilder &launch_ctx);

  LaunchContextBuilder make_launch_context();

  int insert_arg(DataType dt, bool is_nparray);

  int insert_ret(DataType dt);

  float64 get_ret_float(int i);

  int64 get_ret_int(int i);

  void set_arch(Arch arch);

  void account_for_offloaded(OffloadedStmt *stmt);
};

TLANG_NAMESPACE_END
