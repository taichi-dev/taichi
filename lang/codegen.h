#pragma once

#include "util.h"
#include "visitor.h"
#include "expr.h"

namespace taichi::Tlang {


class CodeGenBase : public Visitor {
public:
  int var_count;
  std::string code, code_suffix;
  std::map<NodeType, std::string> binary_ops;
  std::string folder;
  std::string func_name;
  int num_groups;
  int id;
  std::string suffix;

  static int get_code_gen_id() {
    static int id = 0;
    TC_ASSERT(id < 10000);
    return id++;
  }

  CodeGenBase() : Visitor(Visitor::Order::child_first) {
    code = "";
    id = get_code_gen_id();
    func_name = fmt::format("func{:06d}", id);
    binary_ops[NodeType::add] = "+";
    binary_ops[NodeType::sub] = "-";
    binary_ops[NodeType::mul] = "*";
    binary_ops[NodeType::div] = "/";
    binary_ops[NodeType::mod] = "%";
    folder = "_tlang_cache/";
    create_directories(folder);
    var_count = 0;
    code_suffix = "\n";
  }

  std::string create_variable() {
    TC_ASSERT(var_count < 10000);
    return fmt::format("var_{:04d}", var_count++);
  }

  std::string get_scalar_suffix(int i) {
    return fmt::format("_{:03d}", i);
  }

  std::string get_source_fn() {
    return fmt::format("{}/tmp{:04d}.{}", folder, id, suffix);
  }

  std::string get_project_fn() {
    return fmt::format("{}/projects/taichi_lang/", get_repo_dir());
  }

  std::string get_library_fn() {
#if defined(TC_PLATFORM_OSX)
    // Note: use .so here will lead to wired behavior...
    return fmt::format("{}/tmp{:04d}.dylib", folder, id);
#else
    return fmt::format("{}/tmp{:04d}.so", folder, id);
#endif
  }

  template <typename... Args>
  void emit_code(std::string f, Args &&... args) {
    if (sizeof...(args)) {
      code += fmt::format(f, std::forward<Args>(args)...) + code_suffix;
    } else {
      code += f + code_suffix;
    }
  }

  void write_code_to_file() {
    {
      std::ofstream of(get_source_fn());
      of << code;
    }
    trash(std::system(
        fmt::format("cp {} {}_unformated", get_source_fn(), get_source_fn())
            .c_str()));
    auto format_ret =
        std::system(fmt::format("clang-format -i {}", get_source_fn()).c_str());
    trash(format_ret);
  }

  FunctionType load_function() {
    auto dll = dlopen(("./" + get_library_fn()).c_str(), RTLD_LAZY);
    TC_ASSERT(dll != nullptr);
    auto ret = dlsym(dll, func_name.c_str());
    TC_ASSERT(ret != nullptr);
    return (FunctionType)ret;
  }
};



}
