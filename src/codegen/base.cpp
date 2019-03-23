#include "base.h"

TLANG_NAMESPACE_BEGIN

std::string CodeGenBase::get_source_path() {
  return fmt::format("{}/{}/{}", get_project_fn(), folder, get_source_name());
}

TLANG_NAMESPACE_END
