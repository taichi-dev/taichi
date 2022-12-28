#include <cstdlib>

inline bool is_ci() {
  char *res = std::getenv("TI_CI");
  if (res == nullptr)
    return false;
  return std::stoi(res);
}
