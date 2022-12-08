#pragma once

#include "taichi/util/lang_util.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/ir.h"
#include "taichi/rhi/arch.h"
#include "taichi/program/callable.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"
#include "taichi/aot/graph_data.h"

namespace taichi::lang {

class Program;

// Refactor2023:FIXME: Move to KernelLauncher
class KernelLaunchContext {
 public:
  KernelLaunchContext(Kernel *kernel, RuntimeContext *ctx);
  explicit KernelLaunchContext(Kernel *kernel);

  KernelLaunchContext(KernelLaunchContext &&) = default;
  KernelLaunchContext &operator=(KernelLaunchContext &&) = default;
  KernelLaunchContext(const KernelLaunchContext &) = delete;
  KernelLaunchContext &operator=(const KernelLaunchContext &) = delete;

  void set_arg_float(int arg_id, float64 d);

  // Created signed and unsigned version for argument range check of pybind
  void set_arg_int(int arg_id, int64 d);
  void set_arg_uint(int arg_id, uint64 d);

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

  uint64 get_ret_raw(Device *device, unsigned retNo) const;

  template <typename T>
  T get_ret(Device *device, unsigned retNo) const;

  float64 get_ret_float(Device *device, unsigned retNo) const;
  int64 get_ret_int(Device *device, unsigned retNo) const;
  uint64 get_ret_uint(Device *device, unsigned retNo) const;
  std::vector<int64> get_ret_int_tensor(Device *device, unsigned retNo) const;
  std::vector<uint64> get_ret_uint_tensor(Device *device, unsigned retNo) const;
  std::vector<float64> get_ret_float_tensor(Device *device,
                                            unsigned retNo) const;

  RuntimeContext &get_context();

 private:
  template <typename T>
  static T fetch_ret(DataType dt,
                     unsigned retNo,
                     Device *device,
                     RuntimeContext *rt_ctx);

  Kernel *kernel_;
  std::unique_ptr<RuntimeContext> owned_ctx_;
  // |ctx_| *almost* always points to |owned_ctx_|. However, it is possible
  // that the caller passes a RuntimeContext pointer externally. In that case,
  // |owned_ctx_| will be nullptr.
  // Invariant: |ctx_| will never be nullptr.
  RuntimeContext *ctx_;
};

class TI_DLL_EXPORT Kernel : public Callable {
 public:
  std::string name;
  std::vector<SNode *> no_activate;

  bool is_accessor{false};
  bool is_evaluator{false};
  AutodiffMode autodiff_mode{AutodiffMode::kNone};

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

  bool ir_is_ast() const {
    return ir_is_ast_;
  }

  bool lowered() const {
    return lowered_;
  }

  void set_lowered(bool lowered) const {
    lowered_ = lowered;
  }

  // Refactor2023:FIXME: Move to KernelLauncher
  KernelLaunchContext make_launch_context();

  // Refactor2023:FIXME: Pre-refactor & Remove
  uint64 get_next_task_id() {
    return task_counter_++;
  }

  void mark_as_from_cache() {
    from_cache_ = true;
  }

  [[nodiscard]] std::string get_name() const override;
  /**
   * Whether the given |arch| is supported in the lower() method.
   *
   * @param arch: The arch to check
   * @return: True if supported.
   */
  // Refactor2023:FIXME: Remove
  static bool supports_lowering(Arch arch);

  void set_kernel_key_for_cache(const std::string &kernel_key) {
    kernel_key_ = kernel_key;
  }

  const std::string &get_cached_kernel_key() {
    return kernel_key_;
  }

  // Refactor2023:FIXME: Remove
  void offload_to_executable(const CompileConfig &config, IRNode *stmt);

  // Refactor2023:FIXME: Remove
  FunctionType get_compiled_func() {
    return compiled_;
  }

  void set_compiled_func(FunctionType func) {
    compiled_ = func;
  }

 private:
  void init(Program &program,
            const std::function<void()> &func,
            const std::string &name = "",
            AutodiffMode autodiff_mode = AutodiffMode::kNone);

  // True if |ir| is a frontend AST. False if it's already offloaded to CHI IR.
  bool ir_is_ast_{false};
  // The closure that, if invoked, launches the backend kernel (shader)
  // Refactor2023:FIXME: Remove
  FunctionType compiled_{nullptr};
  // A flag to record whether |ir| has been fully lowered.
  // lower initial AST all the way down to a bunch of
  // OffloadedStmt for async execution TODO(Lin): Check this comment
  mutable bool lowered_{false};
  std::atomic<uint64> task_counter_{0};
  std::string kernel_key_;
  bool from_cache_{false};
};

// Refactor2023:FIXME: Remove
void launch_kernel(Program *prog,
                   const CompileConfig &compiple_config,
                   Kernel &kernel,
                   RuntimeContext &ctx);

template <typename T>
T KernelLaunchContext::get_ret(Device *device, unsigned retNo) const {
  auto *dt = kernel_->rets[retNo].dt->get_compute_type();
  return fetch_ret<float64>(dt, retNo, device, ctx_);
}

}  // namespace taichi::lang
