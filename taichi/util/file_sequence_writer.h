#pragma once

#include "taichi/util/lang_util.h"
#ifdef TI_WITH_LLVM
#include "taichi/runtime/llvm/llvm_fwd.h"
#endif

namespace taichi {
namespace lang {
class IRNode;
}
}  // namespace taichi

namespace taichi {

class FileSequenceWriter {
 public:
  FileSequenceWriter(std::string filename_template, std::string file_type);

#ifdef TI_WITH_LLVM
  // returns filename
  std::string write(llvm::Module *module);
#endif

  std::string write(lang::IRNode *irnode);

  std::string write(const std::string &str);

 private:
  int counter_;
  std::string filename_template_;
  std::string file_type_;

  std::pair<std::ofstream, std::string> create_new_file();
};

}  // namespace taichi
