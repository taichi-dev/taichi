#pragma once

#include "taichi/common/core.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/system/dynamic_loader.h"
#include "taichi/util/action_recorder.h"
#include "struct_cc.h"
#include "cc_program.h"
#include "cc_runtime.h"
#include "cc_kernel.h"
#include "cc_layout.h"
#include "cc_utils.h"
#include "codegen_cc.h"
#include "context.h"
#include "taichi/lang_util.h"
#include <vector>
#include <memory>

TI_NAMESPACE_BEGIN
class DynamicLoader;
TI_NAMESPACE_END

TLANG_NAMESPACE_BEGIN

class CCKernel;

class CCProgramImpl : public ProgramImpl {
public:
    CCProgramImpl(CompileConfig &config) : ProgramImpl(config) {
    }
    FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) override;

    std::size_t get_snode_num_dynamically_allocated(
            SNode *snode,
            uint64 *result_buffer) override {
        return 0;  // TODO: support sparse in vulkan
    }

    void materialize_runtime(MemoryPool *memory_pool,
                             KernelProfilerBase *profiler,
                             uint64 **result_buffer_ptr) override;

    void materialize_snode_tree(SNodeTree *tree,
                                std::vector<std::unique_ptr<SNodeTree>> &,
                                std::unordered_map<int, SNode *> &,
                                uint64 *result_buffer) override;

    void synchronize() override {
        return;
    }

    std::unique_ptr<AotModuleBuilder> make_aot_module_builder() override {
        return nullptr;
    }

    virtual void destroy_snode_tree(SNodeTree *snode_tree) override {
        return;
    }

    ~CCProgramImpl() {
    }

private:
  void add_kernel(std::unique_ptr<cccp::CCKernel> kernel);

  std::vector<std::unique_ptr<cccp::CCKernel>> kernels;
  bool need_relink{true};

};

namespace cccp {

class CCKernel;
class CCLayout;
class CCRuntime;
struct CCContext;

using CCFuncEntryType = void(CCContext *);

class CCProgram {
  // Launch C compiler to compile generated source code, and run them
 public:
  CCProgram(Program *program);
  ~CCProgram();

  void add_kernel(std::unique_ptr<CCKernel> kernel);
  CCFuncEntryType *load_kernel(std::string const &name);
  void compile_layout(SNode *root);
  void init_runtime();
  void relink();

  CCLayout *get_layout() {
    return layout.get();
  }

  CCRuntime *get_runtime() {
    return runtime.get();
  }

  CCContext *update_context(Context *ctx);
  void context_to_result_buffer();

  Program *const program;

 private:
  std::vector<char> args_buf;
  std::vector<char> root_buf;
  std::vector<char> gtmp_buf;
  std::vector<std::unique_ptr<CCKernel>> kernels;
  std::unique_ptr<CCContext> context;
  std::unique_ptr<CCRuntime> runtime;
  std::unique_ptr<CCLayout> layout;
  std::unique_ptr<DynamicLoader> dll;
  std::string dll_path;
  bool need_relink{true};
};

}  // namespace cccp
TLANG_NAMESPACE_END
