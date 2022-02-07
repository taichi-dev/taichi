#include "taichi/util/action_recorder.h"
#include "taichi/util/str.h"

TI_NAMESPACE_BEGIN

void ActionArg::serialize(std::ostream &ss) const {
  ss << key << ": ";
  if (type == argument_type::str) {
    ss << lang::c_quoted(val_str);
  } else if (type == argument_type::int64) {
    ss << std::to_string(val_int64);
  } else {
    ss << std::to_string(val_float64);
  }
}

ActionRecorder &ActionRecorder::get_instance() {
  static ActionRecorder rec;
  return rec;
}

ActionRecorder::ActionRecorder() {
}

void ActionRecorder::start_recording(const std::string &fn) {
  TI_INFO("ActionRecorder: start recording to [{}]", fn);
  TI_ASSERT(!running_);
  running_ = true;
  ofs_.open(fn);
}

void ActionRecorder::stop_recording() {
  TI_INFO("ActionRecorder: stop recording");
  TI_ASSERT(running_);
  running_ = false;
  ofs_.close();
}

bool ActionRecorder::is_recording() {
  return running_;
}

void ActionRecorder::record(const std::string &content,
                            const std::vector<ActionArg> &arguments) {
  if (!running_)
    return;
  ofs_ << "- action: \"" << content << "\"" << std::endl;
  for (auto &arg : arguments) {
    ofs_ << "  ";
    arg.serialize(ofs_);
    ofs_ << std::endl;
  }
  ofs_.flush();
}

TI_NAMESPACE_END
