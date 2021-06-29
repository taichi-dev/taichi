#pragma once

#include <memory>

#include "taichi/program/program.h"

namespace taichi {
namespace lang {

class TestProgram {
 public:
  void setup();

  Program *prog() {
    return prog_.get();
  }

 private:
  std::unique_ptr<Program> prog_{nullptr};
};

}  // namespace lang
}  // namespace taichi
