#pragma once

#include <string>

namespace taichi {
namespace lang {

class Kernel;
class DataType;

class AotModuleBuilder {
 public:
  virtual ~AotModuleBuilder() = default;

  void add(const std::string &identifier, Kernel *kernel);

  void add_field(const std::string &identifier,
<<<<<<< HEAD
                 bool is_scalar,
=======
                 bool is_vector,
>>>>>>> 57c1a18f (Auto Format)
                 DataType dt,
                 std::pair<int, int> shape,
                 int vector_size);

  void add_kernel_template(const std::string &identifier,
                           const std::string &key,
                           Kernel *kernel);

  virtual void dump(const std::string &output_dir,
                    const std::string &filename) const = 0;

 protected:
  /**
   * Intended to be overriden by each backend's implementation.
   */
  virtual void add_per_backend(const std::string &identifier,
                               Kernel *kernel) = 0;
  virtual void add_per_backend_field(const std::string &identifier,
<<<<<<< HEAD
                                     bool is_scalar,
=======
                                     bool is_vector,
>>>>>>> 57c1a18f (Auto Format)
                                     DataType dt,
                                     std::pair<int, int> shape,
                                     int vector_size) = 0;
  virtual void add_per_backend_tmpl(const std::string &identifier,
                                    const std::string &key,
                                    Kernel *kernel) = 0;
};

}  // namespace lang
}  // namespace taichi
