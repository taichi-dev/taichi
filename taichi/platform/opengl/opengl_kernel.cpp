#include "opengl_kernel.h"
#include "opengl_api.h"

#include <taichi/backends/base.h>
#include <taichi/backends/kernel.h>

TLANG_NAMESPACE_BEGIN
namespace opengl {

SSBO::SSBO(size_t data_size_)
  : data(std::calloc(data_size_, 1)), data_size(data_size_)
{}

void SSBO::load_from(const void *buffer)
{
  std::memcpy(data, buffer, data_size);
}

void SSBO::save_to(void *buffer)
{
  std::memcpy(buffer, data, data_size);
}

SSBO::~SSBO()
{
  std::free(data);
}

}  // namespace opengl
TLANG_NAMESPACE_END
