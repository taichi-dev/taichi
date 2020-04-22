#include "statistics.h"

TI_NAMESPACE_BEGIN

Statistics stat;

void Statistics::add(std::string key, Statistics::value_type value) {
  counters[key] += value;
}

void Statistics::print(std::string *output) {
  std::vector<std::string> keys;
  for (auto const &item : counters)
    keys.push_back(item.first);

  std::sort(keys.begin(), keys.end());

  std::stringstream ss;
  for (auto const &k : keys)
    ss << fmt::format("{:20}: {:.2f}\n", k, counters[k]);

  if (output) {
    *output = ss.str();
  } else {
    fmt::print(ss.str());
  }
}

void Statistics::clear() {
  counters.clear();
}

TI_NAMESPACE_END
