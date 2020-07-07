#pragma once

#include <fstream>

#include "taichi/common/core.h"

TI_NAMESPACE_BEGIN

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

  std::string serialize() const;

  std::string key;

  std::string val_str;
  int64 val_int64;
  float64 val_float64;

  enum argument_type { str, int64, float64 };
  argument_type type;
};

class ActionRecorder {
 public:
  static ActionRecorder &get_instance();

  static void record(const std::string &content,
                     const std::vector<ActionArg> &arguments = {});

  static void start_recording(const std::string &fn);

  static void stop_recording();

 private:
  ActionRecorder();

  void record_(const std::string &content,
               const std::vector<ActionArg> &arguments);

  void start_recording_(const std::string &fn);

  std::ofstream ofs;

  bool running{false};
};

TI_NAMESPACE_END
