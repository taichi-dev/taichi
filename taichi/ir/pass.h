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

class PassManager;

class Pass {
 public:
  static const PassID id;
  class Result {
   public:
    Result() = default;
  };
};
const PassID Pass::id = "undefined";

class PassManager {
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
