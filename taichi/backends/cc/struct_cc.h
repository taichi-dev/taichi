#pragma once

#include "taichi/lang_util.h"
#include "taichi/util/line_appender.h"
#include "taichi/ir/snode.h"

TLANG_NAMESPACE_BEGIN
class CCProgramImpl;
namespace cccp {

class CCLayout;

class CCLayoutGen {
  // Generate corresponding C Source Code for Taichi Structures
 public:
  CCLayoutGen(CCProgramImpl *cc_program_impl, SNode *root)
      : cc_program_impl_(cc_program_impl), root(root) {
  }

  std::unique_ptr<CCLayout> compile();

 private:
  void generate_children(SNode *snode);
  void generate_types(SNode *snode);

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    line_appender.append(std::move(f), std::move(args)...);
  }

  CCProgramImpl *cc_program_impl_;

  SNode *root;
  std::vector<SNode *> snodes;
  LineAppender line_appender;
};

}  // namespace cccp
TLANG_NAMESPACE_END
