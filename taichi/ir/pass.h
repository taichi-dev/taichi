#pragma once

#include "taichi/ir/ir.h"
#include "taichi/program/compile_config.h"

#include <unordered_map>
#include <typeindex>

namespace taichi {
namespace lang {

using PassID = std::string;

class PassContext {
 public:
  PassContext() {
  }
  explicit PassContext(const CompileConfig &config) : config_(config) {
  }
  [[nodiscard]] const CompileConfig &get_config() const {
    return config_;
  }

  // private:
  CompileConfig config_;
};

class AnalysisManager;

class Pass {
 public:
  static const PassID id;

  // Analysis results.
  class Result {
   public:
    Result() = default;
  };

  // The numbers for the cases are assigned to make sure that Failure & anything
  // is Failure, SuccessWithChange & any success is SuccessWithChange.
  enum class Status {
    Failure = 0x00,
    SuccessWithChange = 0x10,
    SuccessWithoutChange = 0x11,
  };

  virtual Status run(const PassContext &ctx,
                     IRNode *module,
                     AnalysisManager *amgr) {
  }
};
const PassID Pass::id = "undefined";

class AnalysisManager {
 public:
  template <typename XPass>
  const typename XPass::Result &get_pass_result() {
    auto result = result_.find(XPass::id);
    TI_ASSERT(result != result_.end());
    return *(result.second);
  }

  template <typename XPass>
  void put_pass_result(std::unique_ptr<typename XPass::Result> result) {
    result_[XPass::id] = std::move(result);
  }

 private:
  std::unordered_map<PassID, std::unique_ptr<Pass::Result>> result_;
};

}  // namespace lang
}  // namespace taichi
