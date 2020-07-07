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

  std::string serialize() const;

  std::string key;

  std::string val_str;
  int64 val_int64;

  enum argument_type { str, int64 };
  argument_type type;
};

class ActionRecorder {
 public:
  static ActionRecorder &get_instance();

  static void record(const std::string &content,
                     const std::vector<ActionArg> &arguments = {});

 private:
  void record_(const std::string &content,
               const std::vector<ActionArg> &arguments);

  ActionRecorder(const std::string &fn);

  std::ofstream ofs;
};

TI_NAMESPACE_END
