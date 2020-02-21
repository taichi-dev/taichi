#include "opengl_kernel.h"
#include "opengl_api.h"

#include <taichi/codegen/codegen.h>

TLANG_NAMESPACE_BEGIN
namespace opengl {

<<<<<<< HEAD
SSBO::SSBO(size_t data_size_)
  : data_(data_size_), data_size(data_size_)
{}

void SSBO::load_from(const void *buffer)
{
  std::memcpy(data(), buffer, data_size);
}

void SSBO::save_to(void *buffer)
{
  std::memcpy(buffer, data(), data_size);
}

}  // namespace opengl
TLANG_NAMESPACE_END
