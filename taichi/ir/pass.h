#pragma once

#include "taichi/ir/ir.h"
#include "taichi/program/compile_config.h"

#include <unordered_map>
#include <typeindex>
#include <utility>

namespace taichi {
namespace lang {

using PassID = std::string;

/*class PassContext {
 public:
  PassContext() {
  }
  explicit PassContext(CompileConfig config) : config_(std::move(config)) {
  }
  [[nodiscard]] const CompileConfig &get_config() const {
    return config_;
  }

 private:
  CompileConfig config_;
};*/

class AnalysisManager;

// Abstract concept of an analysis result.
struct AnalysisResultConcept {
  virtual ~AnalysisResultConcept() = default;
};

template <typename ResultT>
struct AnalysisResultModel : public AnalysisResultConcept {
  explicit AnalysisResultModel(ResultT result) : result(std::move(result)) {
  }
  ResultT result;
};

class Pass {
 public:
  static const PassID id;

  // The numbers for the cases are assigned to make sure that Failure & anything
  // is Failure, SuccessWithChange & any success is SuccessWithChange.
  enum class Status {
    Failure = 0x00,
    SuccessWithChange = 0x10,
    SuccessWithoutChange = 0x11,
  };

  /*virtual Status run(const PassContext &ctx,
                     IRNode *module,
                     AnalysisManager *amgr) {
    return Status::Failure;
  }*/

  virtual ~Pass() = default;
};

class AnalysisManager {
 public:
  template <typename PassT>
  typename PassT::Result *get_pass_result() {
    auto result = result_.find(PassT::id);
    if (result == result_.end()) {
      return nullptr;
    }
    using ResultModelT = AnalysisResultModel<typename PassT::Result>;
    return &(static_cast<ResultModelT *>(result->second.get())->result);
  }

  template <typename PassT>
  void put_pass_result(typename PassT::Result &&result) {
    using ResultModelT = AnalysisResultModel<typename PassT::Result>;
    result_[PassT::id] = std::make_unique<ResultModelT>(std::move(result));
  }

 private:
  std::unordered_map<PassID, std::unique_ptr<AnalysisResultConcept>> result_;
};

}  // namespace lang
}  // namespace taichi
