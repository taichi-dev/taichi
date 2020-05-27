#include <unordered_map>

#include "taichi/common/core.h"

TI_NAMESPACE_BEGIN

class Statistics {
  using value_type = float64;

 private:
  std::unordered_map<std::string, value_type> counters;

 public:
  Statistics() = default;

  void add(std::string key, value_type value = value_type(1));

  void print(std::string *output = nullptr);

  void clear();
};

extern Statistics stat;

TI_NAMESPACE_END
