#pragma once

#include "taichi/common/core.h"
#include "taichi/program/arch.h"
#include "taichi/program/profiler.h"

#include <map>
#include <memory>
#include <functional>

TLANG_NAMESPACE_BEGIN

class Runtime {
 protected:
  ProfilerBase *profiler;

 public:
  Runtime() : profiler(nullptr) {
  }

  // Does the machine really have the corresponding hardware?
  virtual bool detected() = 0;

  void set_profiler(ProfilerBase *profiler) {
    this->profiler = profiler;
  }

  virtual std::size_t get_total_memory() = 0;

  virtual std::size_t get_available_memory() = 0;

  virtual ~Runtime() {
  }

  using Factories = std::map<Arch, std::function<std::unique_ptr<Runtime>()>>;

  static Factories &get_factories() {
    static Factories factories;
    return factories;
  }

  template <typename RuntimeT>
  static void register_impl(Arch arch);

  static std::unique_ptr<Runtime> create(Arch arch);
};

template <typename RuntimeT>
void Runtime::register_impl(Arch arch) {
  auto &factories = get_factories();
  TI_ASSERT(factories.find(arch) == factories.end());
  factories[arch] = [] { return std::make_unique<RuntimeT>(); };
}

TLANG_NAMESPACE_END
