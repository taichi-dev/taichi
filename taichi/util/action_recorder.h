#pragma once

#include <fstream>

#include "taichi/common/core.h"

namespace taichi {

struct ActionArg {
  ActionArg(const std::string &key, const std::string &val)
      : key(key), val_str(val), type(argument_type::str) {
  }

  ActionArg(const std::string &key, int64 val)
      : key(key), val_int64(val), type(argument_type::int64) {
  }

  ActionArg(const std::string &key, float64 val)
      : key(key), val_float64(val), type(argument_type::float64) {
  }

  ActionArg(const std::string &key, int32 val)
      : key(key), val_int64(val), type(argument_type::int64) {
  }

  ActionArg(const std::string &key, float32 val)
      : key(key), val_float64(val), type(argument_type::float64) {
  }

  void serialize(std::ostream &ss) const;

  std::string key;

  std::string val_str;
  int64 val_int64;
  float64 val_float64;

  enum class argument_type { str, int64, float64 };
  argument_type type;
};

// TODO: Make this thread safe when switching to async mode.
class ActionRecorder {
 public:
  static ActionRecorder &get_instance();

  void record(const std::string &content,
              const std::vector<ActionArg> &arguments = {});

  void start_recording(const std::string &fn);

  void stop_recording();

  bool is_recording();

 private:
  ActionRecorder();

  std::ofstream ofs_;

  bool running_{false};
};

}  // namespace taichi
