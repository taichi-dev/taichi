#include "taichi/util/action_recorder.h"
#include "taichi/util/str.h"

TI_NAMESPACE_BEGIN

std::string ActionArg::serialize() const {
  std::string ret = key + ": ";
  if (type == argument_type::str) {
    ret += lang::c_quoted(val_str);
  } else if (type == argument_type::int64) {
    ret += std::to_string(val_int64);
  } else {
    ret += std::to_string(val_float64);
  }
  return ret;
}

ActionRecorder &ActionRecorder::get_instance() {
  static ActionRecorder rec;
  return rec;
}

ActionRecorder::ActionRecorder() {
}

void ActionRecorder::start_recording(const std::string &fn) {
  TI_INFO("ActionRecorder: start recording to [{}]", fn);
  TI_ASSERT(!running);
  running = true;
  ofs.open(fn);
}

void ActionRecorder::stop_recording() {
  TI_INFO("ActionRecorder: stop recording");
  TI_ASSERT(running);
  running = false;
  ofs.close();
}

bool ActionRecorder::is_recording() {
  return running;
}

void ActionRecorder::record(const std::string &content,
                            const std::vector<ActionArg> &arguments) {
  if (!running)
    return;
  ofs << "- action: \"" << content << "\"" << std::endl;
  for (auto &arg : arguments) {
    ofs << "  " << arg.serialize() << std::endl;
  }
  ofs.flush();
}

TI_NAMESPACE_END
