#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "taichi/util/file_sequence_writer.h"


TLANG_NAMESPACE_BEGIN

FileSequenceWriter::FileSequenceWriter(std::string filename_template,
                                                     std::string file_type)
    : counter(0), filename_template(filename_template), file_type(file_type) {
}

void FileSequenceWriter::write(llvm::Module *module) {
  std::string str;
  llvm::raw_string_ostream ros(str);
  module->print(ros, nullptr);
  write(str);
}

void FileSequenceWriter::write(const std::string &str) {
  create_new_file() << str;
}

void FileSequenceWriter::write(IRNode *irnode) {
  std::string content;
  irpass::print(irnode, &content);
  write(content);
}

std::ofstream FileSequenceWriter::create_new_file() {
  auto fn = fmt::format(filename_template, counter);
  TI_INFO("Saving {} to {}", file_type, fn);
  counter++;
  return std::ofstream(fn);
}

TLANG_NAMESPACE_END
