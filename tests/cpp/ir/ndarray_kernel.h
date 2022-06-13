#pragma once
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/inc/constants.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

std::unique_ptr<Kernel> setup_kernel1(Program *prog);

std::unique_ptr<Kernel> setup_kernel2(Program *prog);
}  // namespace lang
}  // namespace taichi
