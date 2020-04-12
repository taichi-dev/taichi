#include <unordered_map>

#include "taichi/common/util.h"

TI_NAMESPACE_BEGIN

class Statistics {
  using value_type = float64;

 private:
  std::unordered_map<std::string, value_type> counters;

 public:
  Statistics() = default;

  void add(std::string key, value_type value = value_type(1));

  void print(std::string *output = nullptr);
};

extern Statistics stat;

TI_NAMESPACE_END
