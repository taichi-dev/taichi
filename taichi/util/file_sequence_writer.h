#pragma once

#include "taichi/lang_util.h"
#include "taichi/ir/transforms.h"
#include "taichi/llvm/llvm_fwd.h"

TLANG_NAMESPACE_BEGIN

class FileSequenceWriter {
 public:
  FileSequenceWriter(std::string filename_template, std::string file_type);

  void write(llvm::Module *module);

  void write(IRNode *irnode);

  void write(const std::string &str);

 private:
  int counter;
  std::string filename_template;
  std::string file_type;

  std::ofstream create_new_file();
};

TLANG_NAMESPACE_END
