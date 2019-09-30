// The code generator base class

#include "base.h"
#include <xxhash.h>
#include <sstream>
#include <taichi/system/timer.h>

TLANG_NAMESPACE_BEGIN

std::string CodeGenBase::get_source_path() {
  return fmt::format("{}/{}", folder, get_source_name());
}

std::string CodeGenBase::get_library_path() {
#if defined(TC_PLATFORM_OSX)
  // Note: use .so here will lead to wired behavior...
  return fmt::format("{}/tmp{:04d}.dylib", folder, id);
#else
  return fmt::format("{}/{}.so", folder, get_source_name());
#endif
}

void CodeGenBase::write_source() {
  std::ifstream ifs(get_source_path());
  if (ifs) {
    std::string firstline;
    std::getline(ifs, firstline);
    if (firstline.find("debug") != firstline.npos) {
      TC_WARN("Debugging file {}. Code overridden.", get_source_path());
      return;
    }
  }
  {
    std::ofstream of(get_source_path());
    for (auto const &k : codes) {
      of << "// region " << get_region_name(k.first) << std::endl;
      of << k.second;
    }
  }
}

std::string CodeGenBase::get_source() {
  TC_NOT_IMPLEMENTED
  return "";
}

void CodeGenBase::load_dll() {
  dll = dlopen((get_library_path()).c_str(), RTLD_LAZY);
  if (dll == nullptr) {
    TC_ERROR("{}", dlerror());
  }
  TC_ASSERT(dll != nullptr);
}

void CodeGenBase::disassemble() {
#if defined(TC_PLATFORM_LINUX)
  auto objdump_ret = system(fmt::format("objdump {} -d > {}.s",
                                        get_library_path(), get_library_path())
                                .c_str());
  trash(objdump_ret);
#endif
}

FunctionType CodeGenBase::load_function() {
  return load_function<FunctionType>(func_name);
}

std::string CodeGenBase::get_source_name() {
  return fmt::format("tmp{:04d}.{}", id, suffix);
}

void CodeGenBase::generate_binary(std::string extra_flags) {
  auto t = Time::get_time();
  write_source();
  auto format_ret =
      std::system(fmt::format("clang-format -i {}", get_source_path()).c_str());
  trash(format_ret);
  auto pp_fn = get_source_path() + ".i";
  auto preprocess_cmd = get_current_program().config.preprocess_cmd(
      get_source_path(), pp_fn, extra_flags);
  auto ret = std::system(preprocess_cmd.c_str());
  if (ret) {
    trash(std::system(
        get_current_program()
            .config.preprocess_cmd(get_source_path(), pp_fn, extra_flags, true)
            .c_str()));
    TC_ERROR("Preprocessing failed.");
  }
  std::ifstream ifs(pp_fn);
  TC_ASSERT(ifs);
  auto hash_input =
      preprocess_cmd + std::string(std::istreambuf_iterator<char>(ifs),
                                   std::istreambuf_iterator<char>());
  auto hash = XXH64(hash_input.data(), hash_input.size(), 0);

  std::string cached_binary_fn = db_folder() + fmt::format("/{}.so", hash);
  std::ifstream key_file(cached_binary_fn);
  if (key_file) {
    trash(std::system(
        fmt::format("cp {} {}", cached_binary_fn, get_library_path()).c_str()));
  } else {
    /*
    trash(std::system(
        fmt::format("cp {} {}", get_source_path(), get_source_path())
            .c_str()));
            */
    auto cmd = get_current_program().config.compile_cmd(
        get_source_path(), get_library_path(), extra_flags);
    auto compile_ret = std::system(cmd.c_str());
    if (compile_ret != 0) {
      TC_WARN("Compilation cmd: {}", cmd);
      auto cmd = get_current_program().config.compile_cmd(
          get_source_path(), get_library_path(), extra_flags, true);
      trash(std::system(cmd.c_str()));
      TC_ERROR("Source {} compilation failed.", get_source_path());
    } else {
      trash(std::system(
          fmt::format("cp {} {}", get_library_path(), cached_binary_fn)
              .c_str()));
    }
  }
  trash(std::system(fmt::format("rm {}", pp_fn).c_str()));
  TC_INFO("Compilation time: {:.1f} ms", 1000 * (Time::get_time() - t));
}

CodeGenBase::~CodeGenBase() {
}

TLANG_NAMESPACE_END
