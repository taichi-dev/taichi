#pragma once

#include "taichi/util/lang_util.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/ir.h"
#include "taichi/rhi/arch.h"
#include "taichi/program/callable.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"
#include "taichi/aot/graph_data.h"
#include "taichi/program/launch_context_builder.h"

namespace taichi::lang {

class Program;

class TI_DLL_EXPORT Kernel : public Callable {
 public:
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

  void set_lowered(bool lowered) {
    lowered_ = lowered;
  }

  void compile(const CompileConfig &compile_config);

  void operator()(const CompileConfig &compile_config,
                  LaunchContextBuilder &ctx_builder);

  LaunchContextBuilder make_launch_context();

  template <typename T>
  T fetch_ret(DataType dt, int i);

  float64 get_ret_float(int i);
  int64 get_ret_int(int i);
  uint64 get_ret_uint(int i);
  std::vector<int64> get_ret_int_tensor(int i);
  std::vector<uint64> get_ret_uint_tensor(int i);
  std::vector<float64> get_ret_float_tensor(int i);

  uint64 get_next_task_id() {
    return task_counter_++;
  }

  [[nodiscard]] std::string get_name() const override;

  void set_kernel_key_for_cache(const std::string &kernel_key) const {
    kernel_key_ = kernel_key;
  }

  const std::string &get_cached_kernel_key() const {
    return kernel_key_;
  }

 private:
  void init(Program &program,
            const std::function<void()> &func,
            const std::string &name = "",
            AutodiffMode autodiff_mode = AutodiffMode::kNone);

  // True if |ir| is a frontend AST. False if it's already offloaded to CHI IR.
  bool ir_is_ast_{false};
  // The closure that, if invoked, launches the backend kernel (shader)
  FunctionType compiled_{nullptr};
  // A flag to record whether |ir| has been fully lowered.
  // lower initial AST all the way down to a bunch of
  // OffloadedStmt for async execution TODO(Lin): Check this comment
  bool lowered_{false};
  std::atomic<uint64> task_counter_{0};
  mutable std::string kernel_key_;
};

}  // namespace taichi::lang
