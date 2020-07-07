#pragma once

#include <fstream>

#include "taichi/common/core.h"

TI_NAMESPACE_BEGIN

class ActionRecorder {
 public:
  std::ofstream ofs;

  int indentation{0};

  ActionRecorder(std::string fn) {
    ofs.open(fn);
    ofs << "Taichi Kernel Action Recorder" << std::endl;
  }

  class IndentGuard {
   public:
    ActionRecorder *rec;
    IndentGuard(ActionRecorder *rec) : rec(rec) {
      rec->indentation += 1;
    }

    ~IndentGuard() {
      rec->indentation -= 1;
    }
  };

  IndentGuard get_indent_guard() {
    return IndentGuard(this);
  }

  void record(std::string content) {
    ofs << "* " + std::string(indentation * 2, ' ') + content << std::endl;
    ofs.flush();
  }
};

inline ActionRecorder &get_action_recorder() {
  static ActionRecorder rec("actions.txt");
  return rec;
}

TI_NAMESPACE_END
