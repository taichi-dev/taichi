#pragma once
#include "../util.h"
#include "../snode.h"
#include "../ir.h"
#include "../program.h"
#include <dlfcn.h>

TLANG_NAMESPACE_BEGIN

// Base class for Struct, CPU, GPU backends
class CodeGenBase {
 public:
  int id;
  void *dll;
  std::string line_suffix;
  std::string folder;
  std::string func_name;
  std::string suffix;

  enum class CodeRegion : int {
    header,
    gpu_kernels,
    body,
  };

  static std::string get_region_name(CodeRegion r) {
    static std::map<CodeRegion, std::string> type_names;
    if (type_names.empty()) {
#define REGISTER_TYPE(i) type_names[CodeRegion::i] = #i;
      REGISTER_TYPE(header);
      REGISTER_TYPE(gpu_kernels);
      REGISTER_TYPE(body);
#undef REGISTER_TYPE
    }
    return type_names[r];
  }

  std::map<CodeRegion, std::string> codes;

  CodeRegion current_code_region;

  class CodeRegionGuard {
    CodeGenBase *codegen;
    CodeRegion previous;

   public:
    CodeRegionGuard(CodeGenBase *codegen, CodeRegion current)
        : codegen(codegen), previous(codegen->current_code_region) {
      codegen->current_code_region = current;
    }

    ~CodeRegionGuard() {
      codegen->current_code_region = previous;
    }
  };

  CodeRegionGuard get_region_guard(CodeRegion cr) {
    return CodeRegionGuard(this, cr);
  }

#define CODE_REGION(region) auto _____ = codegen->get_region_guard(CodeGenBase::CodeRegion::region);
#define CODE_REGION_VAR(region) auto _____ = codegen->get_region_guard(region);

  static int get_kernel_id() {
    static int id = 0;
    TC_ASSERT(id < 10000);
    return id++;
  }

  std::string db_folder() {  // binary database
    return folder + "/db";
  }

  CodeGenBase(const std::string &kernel_name = "") {
    id = get_kernel_id();
    func_name = fmt::format("k{:04d}_{}", id, kernel_name);

    dll = nullptr;
    current_code_region = CodeRegion::header;

    folder = get_project_fn() + "/_tlang_cache/";
    create_directories(folder);
    create_directories(db_folder());
    line_suffix = "\n";
  }

  std::string get_source_name();

  std::string get_source_path();

  std::string get_library_path();

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    if (codes.find(current_code_region) == codes.end()) {
      codes[current_code_region] = "";
    }
    codes[current_code_region] +=
        fmt::format(f, std::forward<Args>(args)...) + line_suffix;
  }

  void write_source();

  std::string get_source();

  void load_dll();

  template <typename T>
  T load_function(std::string name) {
    if (dll == nullptr) {
      load_dll();
    }
    auto ret = dlsym(dll, name.c_str());
    TC_ASSERT(ret != nullptr);
    return (T)ret;
  }

  FunctionType load_function();

  void generate_binary(std::string extra_flags);

  void disassemble();

  ~CodeGenBase();
};

inline std::vector<std::string> indices_str(SNode *snode,
                                            int lane_id,
                                            const std::vector<Stmt *> indices) {
  std::vector<std::string> ret(max_num_indices, "0");
  for (int i = 0; i < (int)indices.size(); i++) {
    if (snode->physical_index_position[i] != -1) {
      ret[snode->physical_index_position[i]] =
          indices[i]->raw_name() +
          (lane_id >= 0 ? fmt::format("[{}]", lane_id) : "");
    }
  }
  return ret;
}

TLANG_NAMESPACE_END
