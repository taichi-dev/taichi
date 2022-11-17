#include "statistics.h"

namespace taichi {

Statistics stat;

void Statistics::add(std::string key, Statistics::value_type value) {
  std::lock_guard<std::mutex> _(counters_map_mutex_);
  counters_[key] += value;
}

void Statistics::print(std::string *output) {
  std::vector<std::string> keys;
  for (auto const &item : counters_)
    keys.push_back(item.first);

  std::sort(keys.begin(), keys.end());

  std::stringstream ss;
  for (auto const &k : keys)
    ss << fmt::format("{:20}: {:.2f}\n", k, counters_[k]);

  if (output) {
    *output = ss.str();
  } else {
    fmt::print(ss.str());
  }
}

void Statistics::clear() {
  counters_.clear();
}

}  // namespace taichi
