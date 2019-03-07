#include "util.h"
#include <dlfcn.h>

TLANG_NAMESPACE_BEGIN

class CodeGenBase {
 public:
  int var_count;
  int snode_count;
  int accessor_count;
  std::string code_suffix;
  std::string folder;
  std::string func_name;
  std::string source_name;
  int num_groups;
  int id;
  std::string suffix;
  void *dll;

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

  static int get_code_gen_id() {
    static int id = 0;
    TC_ASSERT(id < 10000);
    return id++;
  }

  CodeGenBase() {
    id = get_code_gen_id();
    func_name = fmt::format("func{:06d}", id);
    suffix = "cpp";  // TODO: remove this hack
    source_name = fmt::format("tmp{:04d}.{}", id, suffix);

    dll = nullptr;
    current_code_region = CodeRegion::header;

    folder = "_tlang_cache/";
    create_directories(folder);
    var_count = 0;
    snode_count = 0;
    accessor_count = 0;
    code_suffix = "\n";
  }

  template <typename T>
  static std::string vec_to_list(std::vector<T> val, std::string bracket) {
    std::string members = bracket;
    bool first = true;
    for (int i = 0; i < (int)val.size(); i++) {
      if (!first) {
        members += ",";
      }
      first = false;
      members += fmt::format("{}", val[i]);
    }
    if (bracket == "<") {
      members += ">";
    } else if (bracket == "{") {
      members += "}";
    } else if (bracket == "(") {
      members += ")";
    } else if (bracket != "") {
      TC_P(bracket);
      TC_NOT_IMPLEMENTED
    }
    return members;
  }

  std::string create_variable() {
    TC_ASSERT(var_count < 10000);
    return fmt::format("var_{:04d}", var_count++);
  }

  std::string create_snode() {
    TC_ASSERT(snode_count < 10000);
    return fmt::format("SNode{:04d}", snode_count++);
  }

  std::string create_accessor() {
    TC_ASSERT(accessor_count < 10000);
    return fmt::format("access_{:04d}", accessor_count++);
  }

  std::string get_source_fn();

  std::string get_library_fn() {
#if defined(TC_PLATFORM_OSX)
    // Note: use .so here will lead to wired behavior...
    return fmt::format("{}/tmp{:04d}.dylib", folder, id);
#else
    return fmt::format("{}/{}.so", folder, source_name);
#endif
  }

  template <typename... Args>
  void emit_code(std::string f, Args &&... args) {
    emit(f, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    if (codes.find(current_code_region) == codes.end()) {
      codes[current_code_region] = "";
    }
    codes[current_code_region] +=
        fmt::format(f, std::forward<Args>(args)...) + code_suffix;
  }

  void write_code_to_file() {
    std::ifstream ifs(get_source_fn());
    std::string firstline;
    std::getline(ifs, firstline);
    if (firstline.find("debug") != firstline.npos) {
      TC_WARN("Debugging file {}. Code overridden.", get_source_fn());
      return;
    }
    {
      std::ofstream of(get_source_fn());
      for (auto const &k : codes) {
        of << "// region " << get_region_name(k.first) << std::endl;
        of << k.second;
      }
    }
    trash(std::system(
        fmt::format("cp {} {}_unformated", get_source_fn(), get_source_fn())
            .c_str()));
    auto format_ret =
        std::system(fmt::format("clang-format -i {}", get_source_fn()).c_str());
    trash(format_ret);
  }

  void load_dll() {
    dll = dlopen(("./" + get_library_fn()).c_str(), RTLD_LAZY);
    if (dll == nullptr) {
      TC_ERROR("{}", dlerror());
    }
    TC_ASSERT(dll != nullptr);
  }

  template <typename T>
  T load_function(std::string name) {
    if (dll == nullptr) {
      load_dll();
    }
    auto ret = dlsym(dll, name.c_str());
    TC_ASSERT(ret != nullptr);
    return (T)ret;
  }

  FunctionType load_function() {
    return load_function<FunctionType>(func_name);
  }

  void disassemble() {
#if defined(TC_PLATFORM_LINUX)
    auto objdump_ret = system(
        fmt::format("objdump {} -M intel -d > {}.s", get_library_fn(), get_library_fn())
            .c_str());
    trash(objdump_ret);
#endif
  }
};

TLANG_NAMESPACE_END
