#pragma once
#include "../util.h"
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
    exterior_shared_variable_begin,
    exterior_loop_begin,
    interior_shared_variable_begin,
    interior_loop_begin,
    body,
    interior_loop_end,
    residual_begin,
    residual_body,
    residual_end,
    interior_shared_variable_end,
    exterior_loop_end,
    exterior_shared_variable_end,
    tail
  };

  static std::string get_region_name(CodeRegion r) {
    static std::map<CodeRegion, std::string> type_names;
    if (type_names.empty()) {
#define REGISTER_TYPE(i) type_names[CodeRegion::i] = #i;
      REGISTER_TYPE(header);
      REGISTER_TYPE(exterior_shared_variable_begin);
      REGISTER_TYPE(exterior_loop_begin);
      REGISTER_TYPE(interior_shared_variable_begin);
      REGISTER_TYPE(interior_loop_begin);
      REGISTER_TYPE(body);
      REGISTER_TYPE(interior_loop_end);
      REGISTER_TYPE(residual_begin);
      REGISTER_TYPE(residual_body);
      REGISTER_TYPE(residual_end);
      REGISTER_TYPE(interior_shared_variable_end);
      REGISTER_TYPE(exterior_loop_end);
      REGISTER_TYPE(exterior_shared_variable_end);
      REGISTER_TYPE(tail);
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

#define CODE_REGION(region) auto _____ = get_region_guard(CodeRegion::region);
#define CODE_REGION_VAR(region) auto _____ = get_region_guard(region);

  static int get_kernel_id() {
    static int id = 0;
    TC_ASSERT(id < 10000);
    return id++;
  }

  CodeGenBase() {
    id = get_kernel_id();
    func_name = fmt::format("func{:06d}", id);

    dll = nullptr;
    current_code_region = CodeRegion::header;

    folder = "_tlang_cache/";
    create_directories(folder);
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

  void disassemble();
};

TLANG_NAMESPACE_END
