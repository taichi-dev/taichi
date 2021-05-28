#pragma once

#include <string>

namespace taichi {
namespace lang {

class Kernel;

class AotModuleBuilder {
 public:
  virtual ~AotModuleBuilder() = default;

  void add(const std::string &identifier, Kernel *kernel);

  virtual void dump(const std::string &output_dir) const = 0;

 protected:
  /**
   * Intended to be overriden by each backend's implementation.
   */
  virtual void add_per_backend(const std::string &identifier,
                               Kernel *kernel) = 0;
};

}  // namespace lang
}  // namespace taichi
