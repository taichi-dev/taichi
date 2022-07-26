#include <unordered_map>

#include "taichi/common/core.h"

namespace taichi {

class Statistics {
 public:
  using value_type = float64;
  using counters_map = std::unordered_map<std::string, value_type>;

  Statistics() = default;

  void add(std::string key, value_type value = value_type(1));

  void print(std::string *output = nullptr);

  void clear();

  inline const counters_map &get_counters() {
    return counters_;
  }

 private:
  counters_map counters_;
};

extern Statistics stat;

}  // namespace taichi
