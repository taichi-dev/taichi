#include "base.h"

TLANG_NAMESPACE_BEGIN

std::string CodeGenBase::get_source_path() {
  return fmt::format("{}/{}/{}", get_project_fn(), folder, get_source_name());
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
  std::string firstline;
  std::getline(ifs, firstline);
  if (firstline.find("debug") != firstline.npos) {
    TC_WARN("Debugging file {}. Code overridden.", get_source_path());
    return;
  }
  {
    std::ofstream of(get_source_path());
    for (auto const &k : codes) {
      of << "// region " << get_region_name(k.first) << std::endl;
      of << k.second;
    }
  }
  trash(std::system(
      fmt::format("cp {} {}_unformated", get_source_path(), get_source_path())
          .c_str()));
  auto format_ret =
      std::system(fmt::format("clang-format -i {}", get_source_path()).c_str());
  trash(format_ret);
}

void CodeGenBase::load_dll() {
  dll = dlopen(("./" + get_library_path()).c_str(), RTLD_LAZY);
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

TLANG_NAMESPACE_END
