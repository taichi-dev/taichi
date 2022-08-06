#pragma once

#include "taichi/util/lang_util.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/ir.h"
#include "taichi/rhi/arch.h"
#include "taichi/program/callable.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"
#include "taichi/aot/graph_data.h"

TLANG_NAMESPACE_BEGIN

class Program;

class TI_DLL_EXPORT Kernel : public Callable {
 public:
  std::string name;
  std::vector<SNode *> no_activate;
  Arch arch;

  bool is_accessor{false};
  bool is_evaluator{false};
  AutodiffMode autodiff_mode{AutodiffMode::kNone};

  class LaunchContextBuilder {
   public:
    LaunchContextBuilder(Kernel *kernel, RuntimeContext *ctx);
    explicit LaunchContextBuilder(Kernel *kernel);

    LaunchContextBuilder(LaunchContextBuilder &&) = default;
    LaunchContextBuilder &operator=(LaunchContextBuilder &&) = default;
    LaunchContextBuilder(const LaunchContextBuilder &) = delete;
    LaunchContextBuilder &operator=(const LaunchContextBuilder &) = delete;

    void set_arg_float(int arg_id, float64 d);

    void set_arg_int(int arg_id, int64 d);

    void set_extra_arg_int(int i, int j, int32 d);

    void set_arg_external_array_with_shape(int arg_id,
                                           uintptr_t ptr,
                                           uint64 size,
                                           const std::vector<int64> &shape);

    void set_arg_ndarray(int arg_id, const Ndarray &arr);

    void set_arg_texture(int arg_id, const Texture &tex);
    void set_arg_rw_texture(int arg_id, const Texture &tex);

    // Sets the |arg_id|-th arg in the context to the bits stored in |d|.
    // This ignores the underlying kernel's |arg_id|-th arg type.
    void set_arg_raw(int arg_id, uint64 d);

    RuntimeContext &get_context();

   private:
    Kernel *kernel_;
    std::unique_ptr<RuntimeContext> owned_ctx_;
    // |ctx_| *almost* always points to |owned_ctx_|. However, it is possible
    // that the caller passes a RuntimeContext pointer externally. In that case,
    // |owned_ctx_| will be nullptr.
    // Invariant: |ctx_| will never be nullptr.
    RuntimeContext *ctx_;
  };

  Kernel(Program &program,
         const std::function<void()> &func,
         const std::string &name = "",
         AutodiffMode autodiff_mode = AutodiffMode::kNone);

  Kernel(Program &program,
         const std::function<void(Kernel *)> &func,
         const std::string &name = "",
         AutodiffMode autodiff_mode = AutodiffMode::kNone);

  Kernel(Program &program,
         std::unique_ptr<IRNode> &&ir,
         const std::string &name = "",
         AutodiffMode autodiff_mode = AutodiffMode::kNone);

  bool lowered() const {
    return lowered_;
  }

  void compile();

  void compile_to_aot_kernel();

  aot::Kernel *compiled_aot_kernel() {
    return compiled_aot_kernel_.get();
  }

  /**
   * Lowers |ir| to CHI IR level
   *
   * @param to_executable: If true, lowers |ir| to a point where the CHI
   * statements can be directly translated by each backend's codegen.
   */
  void lower(bool to_executable = true);

  void operator()(LaunchContextBuilder &ctx_builder);

  LaunchContextBuilder make_launch_context();

  template <typename T>
  T fetch_ret(DataType dt, int i);

  float64 get_ret_float(int i);

  int64 get_ret_int(int i);

  std::vector<int64> get_ret_int_tensor(int i);

  std::vector<float64> get_ret_float_tensor(int i);

  void set_arch(Arch arch);

  void account_for_offloaded(OffloadedStmt *stmt);

  uint64 get_next_task_id() {
    return task_counter_++;
  }

  void set_from_offline_cache() {
    this->from_offline_cache_ = true;
  }

  [[nodiscard]] std::string get_name() const override;
  /**
   * Whether the given |arch| is supported in the lower() method.
   *
   * @param arch: The arch to check
   * @return: True if supported.
   */
  static bool supports_lowering(Arch arch);

  void set_kernel_key_for_cache(const std::string &kernel_key) {
    kernel_key_ = kernel_key;
  }

  const std::string &get_cached_kernel_key() {
    return kernel_key_;
  }
  void offload_to_executable(IRNode *stmt);

 private:
  void init(Program &program,
            const std::function<void()> &func,
            const std::string &name = "",
            AutodiffMode autodiff_mode = AutodiffMode::kNone);

  // True if |ir| is a frontend AST. False if it's already offloaded to CHI IR.
  bool ir_is_ast_{false};
  // The closure that, if invoked, launches the backend kernel (shader)
  FunctionType compiled_{nullptr};
  // TODO[#5114]: It's kinda redundant to keep both compiled_ (used for JIT
  // execution) as well as compiled_aot_kernel_. In fact we'd better unify
  // everything around compiled_aot_kernel and rename it.
  std::unique_ptr<aot::Kernel> compiled_aot_kernel_{nullptr};
  // A flag to record whether |ir| has been fully lowered.
  // lower initial AST all the way down to a bunch of
  // OffloadedStmt for async execution TODO(Lin): Check this comment
  bool lowered_{false};
  std::atomic<uint64> task_counter_{0};
  std::string kernel_key_;
  bool from_offline_cache_{false};
};

TLANG_NAMESPACE_END
