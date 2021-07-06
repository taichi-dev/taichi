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

<<<<<<< HEAD
  void add_field(const std::string &identifier,
                 bool is_scalar,
                 DataType dt,
                 std::pair<int, int> shape,
                 int vector_size);

  void add_kernel_template(const std::string &identifier,
                           const std::string &key,
=======
  void add_kernel_template(const std::string &identifier, 
                           const std::string &key, 
>>>>>>> c596fb80 (dump metal files ok (txt file to fix))
                           Kernel *kernel);

  virtual void dump(const std::string &output_dir,
                    const std::string &filename) const = 0;

 protected:
  /**
   * Intended to be overriden by each backend's implementation.
   */
  virtual void add_per_backend(const std::string &identifier,
                               Kernel *kernel) = 0;
<<<<<<< HEAD
  virtual void add_per_backend_field(const std::string &identifier,
                                     bool is_scalar,
                                     DataType dt,
                                     std::pair<int, int> shape,
                                     int vector_size) = 0;
  virtual void add_per_backend_tmpl(const std::string &identifier,
                                    const std::string &key,
=======
  virtual void add_per_backend_tmpl(const std::string &identifier, 
                                    const std::string &key, 
>>>>>>> c596fb80 (dump metal files ok (txt file to fix))
                                    Kernel *kernel) = 0;
};

}  // namespace lang
}  // namespace taichi
