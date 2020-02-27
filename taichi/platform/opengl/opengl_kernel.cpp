#include "opengl_kernel.h"
#include "opengl_api.h"

#include <taichi/backends/base.h>
#include <taichi/backends/kernel.h>

TLANG_NAMESPACE_BEGIN
namespace opengl {

SSBO::SSBO(size_t data_size_)
  : data(std::malloc(data_size_)), data_size(data_size_)
{}

void SSBO::load_arguments_from(Context &ctx)
{
  uint64_t *data_i = (uint64_t *)data;
  for (int i = 0; i < taichi_max_num_args; i++) {
    uint64_t value = ctx.get_arg<uint64_t>(i);
    data_i[i] = value;
  }
}

void SSBO::save_returns_to(Context &ctx)
{
  uint64_t *data_i = (uint64_t *)data;
  for (int i = 0; i < taichi_max_num_args; i++) {
    uint64_t value = data_i[i];
    ctx.set_arg<uint64_t>(i, value);
  }
}

void SSBO::update(void *data_r)
{
  std::memcpy(data, data_r, data_size);
}

SSBO::~SSBO()
{
  std::free(data);
}

}  // namespace opengl
TLANG_NAMESPACE_END
