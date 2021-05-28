#pragma once

#include "taichi/lang_util.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/ir.h"
#include "taichi/program/arch.h"
#include "taichi/program/callable.h"

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

TLANG_NAMESPACE_BEGIN

class Program;

class Kernel : public Callable {
 public:
  std::string name;
  std::vector<SNode *> no_activate;
  Arch arch;

  bool is_accessor{false};
  bool is_evaluator{false};
  bool grad{false};

  // TODO: Give "Context" a more specific name.
  class LaunchContextBuilder {
   public:
    LaunchContextBuilder(Kernel *kernel, Context *ctx);
    explicit LaunchContextBuilder(Kernel *kernel);

    LaunchContextBuilder(LaunchContextBuilder &&) = default;
    LaunchContextBuilder &operator=(LaunchContextBuilder &&) = default;
    LaunchContextBuilder(const LaunchContextBuilder &) = delete;
    LaunchContextBuilder &operator=(const LaunchContextBuilder &) = delete;

    void set_arg_float(int arg_id, float64 d);

    void set_arg_int(int arg_id, int64 d);

    void set_extra_arg_int(int i, int j, int32 d);

    void set_arg_external_array(int arg_id, uint64 ptr, uint64 size);

    // Sets the |arg_id|-th arg in the context to the bits stored in |d|.
    // This ignores the underlying kernel's |arg_id|-th arg type.
    void set_arg_raw(int arg_id, uint64 d);

    Context &get_context();

   private:
    Kernel *kernel_;
    std::unique_ptr<Context> owned_ctx_;
    // |ctx_| *almost* always points to |owned_ctx_|. However, it is possible
    // that the caller passes a Context pointer externally. In that case,
    // |owned_ctx_| will be nullptr.
    // Invariant: |ctx_| will never be nullptr.
    Context *ctx_;
  };

  Kernel(Program &program,
         const std::function<void()> &func,
         const std::string &name = "",
         bool grad = false);

  Kernel(Program &program,
         std::unique_ptr<IRNode> &&ir,
         const std::string &name = "",
         bool grad = false);

  bool lowered() const {
    return lowered_;
  }

  void compile();

  /**
   * Lowers |ir| to CHI IR level
   *
   * @param to_executable: If true, lowers |ir| to a point where the CHI
   * statements can be directly translated by each backend's codegen.
   */
  void lower(bool to_executable = true);

  void operator()(LaunchContextBuilder &ctx_builder);

  LaunchContextBuilder make_launch_context();

  float64 get_ret_float(int i);

  int64 get_ret_int(int i);

  void set_arch(Arch arch);

  void account_for_offloaded(OffloadedStmt *stmt);

  [[nodiscard]] std::string get_name() const override;
  /**
   * Whether the given |arch| is supported in the lower() method.
   *
   * @param arch: The arch to check
   * @return: True if supported.
   */
  static bool supports_lowering(Arch arch);

 private:
  // True if |ir| is a frontend AST. False if it's already offloaded to CHI IR.
  bool ir_is_ast_{false};
  // The closure that, if invoked, lauches the backend kernel (shader)
  FunctionType compiled_{nullptr};
  // A flag to record whether |ir| has been fully lowered.
  // lower inital AST all the way down to a bunch of
  // OffloadedStmt for async execution
  bool lowered_{false};
};

TLANG_NAMESPACE_END
