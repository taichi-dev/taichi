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

using namespace taichi::lang::cccp;
using CCFuncEntryType = void(cccp::CCContext *);

class CCProgramImpl : public ProgramImpl {
public:
    explicit CCProgramImpl(CompileConfig &config);

    FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) override;

    std::size_t get_snode_num_dynamically_allocated(
            SNode *snode,
            uint64 *result_buffer) override {
        return 0;  // TODO: support sparse in cc.
    }

    void materialize_runtime(MemoryPool *memory_pool,
                             KernelProfilerBase *profiler,
                             uint64 **result_buffer_ptr) override;

    void materialize_snode_tree(SNodeTree *tree,
                                std::vector<std::unique_ptr<SNodeTree>> &,
                                std::unordered_map<int, SNode *> &,
                                uint64 *result_buffer) override;

    void synchronize() override {
        // Not implemented yet.
    }

    std::unique_ptr<AotModuleBuilder> make_aot_module_builder() override {
        // Not implemented yet.
        return nullptr;
    }

    void destroy_snode_tree(SNodeTree *snode_tree) override {
        // Not implemented yet.
    }

    CCLayout *get_layout() {
        return layout.get();
    }

    CCRuntime *get_runtime() {
        return runtime.get();
    }

    ~CCProgramImpl() {
    }

    CCFuncEntryType *load_kernel(std::string const &name);
    void compile_layout(SNode *root);
    void relink();

    CCContext *update_context(Context *ctx);
    void context_to_result_buffer();

private:
  void add_kernel(std::unique_ptr<CCKernel> kernel);
  void init_runtime();

  std::vector<std::unique_ptr<CCKernel>> kernels;
  std::unique_ptr<CCContext> context;
  std::unique_ptr<CCRuntime> runtime;
  std::unique_ptr<CCLayout> layout;
  std::unique_ptr<DynamicLoader> dll;
  std::string dll_path;
  std::vector<char> args_buf;
  std::vector<char> root_buf;
  std::vector<char> gtmp_buf;
  uint64 *result_buffer;
  bool need_relink{true};
};
TLANG_NAMESPACE_END
