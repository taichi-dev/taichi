#pragma once

#include "taichi/lang_util.h"
#include "taichi/util/line_appender.h"
#include "taichi/ir/snode.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCLayout;

class CCLayoutGen {
  // Generate corresponding C Source Code for Taichi Structures
 public:
  CCLayoutGen(SNode *root) : root(root) {
  }

  std::unique_ptr<CCLayout> compile();

 private:
  void generate_children(SNode *snode);
  void generate_types(SNode *snode);

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    line_appender.append(std::move(f), std::move(args)...);
  }

  SNode *root;
  std::vector<SNode *> snodes;
  LineAppender line_appender;
};

}  // namespace cccp
TLANG_NAMESPACE_END
