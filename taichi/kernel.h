#pragma once

#include "tlang_util.h"
#include "snode.h"
#include "ir.h"

TLANG_NAMESPACE_BEGIN

class Program;

class Kernel {
 public:
  std::unique_ptr<IRNode> ir_holder;
  IRNode *ir;
  Program &program;
  FunctionType compiled;
  std::string name;
  Arch arch;
  struct Arg {
    DataType dt;
    bool is_nparray;
    std::size_t size;
    bool is_return_value;

    Arg(DataType dt = DataType::unknown,
        bool is_nparray = false,
        std::size_t size = 0,
        bool is_return_value = 0)
        : dt(dt),
          is_nparray(is_nparray),
          size(size),
          is_return_value(is_return_value) {
    }
  };
  std::vector<Arg> args;
  bool benchmarking;
  bool is_reduction;  // TODO: systematically treat all types of reduction
  bool grad;

  Kernel(Program &program,
         std::function<void()> func,
         std::string name = "",
         bool grad = false);

  void compile();

  void operator()();

  std::function<void()> func() {
    return std::function<void()>([&] { (*this)(); });
  }

  int insert_arg(DataType dt, bool is_nparray);

  void set_arg_float(int i, float64 d);

  void set_arg_int(int i, int64 d);

  void set_extra_arg_int(int i, int j, int32 d);

  void mark_arg_return_value(int i, bool is_return = true);

  void set_arg_nparray(int i, uint64 ptr, uint64 size);

  void set_arch(Arch arch);
};

TLANG_NAMESPACE_END
