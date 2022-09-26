#pragma once

#include <memory>

#include "taichi/program/program.h"

namespace taichi::lang {

class TestProgram {
 public:
  void setup(Arch arch = Arch::x64);

  Program *prog() {
    return prog_.get();
  }

 private:
  std::unique_ptr<Program> prog_{nullptr};
};

}  // namespace taichi::lang
