#include "taichi/util/action_recorder.h"

TI_NAMESPACE_BEGIN

std::string ActionArg::serialize() const {
  std::string ret = key + ": ";
  if (type == argument_type::str) {
    ret += "\"" + val_str + "\"";
  } else {
    ret += std::to_string(val_int64);
  }
  return ret;
}

ActionRecorder &ActionRecorder::get_instance() {
  static ActionRecorder rec("actions.txt");
  return rec;
}

ActionRecorder::ActionRecorder(const std::string &fn) {
  ofs.open(fn);
}

void ActionRecorder::record_(const std::string &content,
                             const std::vector<ActionArg> &arguments) {
  ofs << "- " << std::endl;
  ofs << "  action: \"" << content << "\"" << std::endl;
  for (auto &arg : arguments) {
    ofs << "  " << arg.serialize() << std::endl;
  }
  ofs.flush();
}

void ActionRecorder::record(const std::string &content,
                            const std::vector<ActionArg> &arguments) {
  get_instance().record_(content, arguments);
}

TI_NAMESPACE_END
