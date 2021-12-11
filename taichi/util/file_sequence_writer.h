#pragma once

#include "taichi/lang_util.h"
#include "taichi/ir/transforms.h"
#ifdef TI_WITH_LLVM
#include "taichi/llvm/llvm_fwd.h"
#endif

TLANG_NAMESPACE_BEGIN

class FileSequenceWriter {
 public:
  FileSequenceWriter(std::string filename_template, std::string file_type);

#ifdef TI_WITH_LLVM
  // returns filename
  std::string write(llvm::Module *module);
#endif

  std::string write(IRNode *irnode);

  std::string write(const std::string &str);

 private:
  int counter_;
  std::string filename_template_;
  std::string file_type_;

  std::pair<std::ofstream, std::string> create_new_file();
};

TLANG_NAMESPACE_END
