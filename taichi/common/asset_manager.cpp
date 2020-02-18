#include "util.h"
#include "asset_manager.h"

TI_NAMESPACE_BEGIN

AssetManager &AssetManager::get_instance() {
  static AssetManager manager;
  return manager;
}

TI_NAMESPACE_END
