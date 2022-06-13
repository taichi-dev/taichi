#pragma once

#include <functional>

namespace taichi {

class RaiiCleanup {
 public:
  using Func = std::function<void()>;

  explicit RaiiCleanup(Func fn);
  ~RaiiCleanup();
  RaiiCleanup(const RaiiCleanup &) = delete;
  RaiiCleanup &operator=(const RaiiCleanup &) = delete;
  RaiiCleanup(RaiiCleanup &&) = default;
  RaiiCleanup &operator=(RaiiCleanup &&) = default;

 private:
  Func fn_;
};

RaiiCleanup make_cleanup(RaiiCleanup::Func fn);

}  // namespace taichi
