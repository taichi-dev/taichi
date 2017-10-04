/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/interface.h>

TC_NAMESPACE_BEGIN

#define TC_IMPLEMENTATION_LOADER(base_class_name, class_name, alias)          \
  class ImplementationLoader_##base_class_name##class_name {                  \
   public:                                                                    \
    ImplementationLoader_##base_class_name##class_name() {                    \
      TC_IMPLEMENTATION_HOLDER_NAME(base_class_name)::get_instance()          \
          ->insert<class_name>(alias);                                        \
    }                                                                         \
    ~ImplementationLoader_##base_class_name##class_name() {                   \
      TC_IMPLEMENTATION_HOLDER_NAME(base_class_name)::get_instance()->remove( \
          alias);                                                             \
    }                                                                         \
  } ImplementationLoader_##base_class_name##class_name##instance;

#define TC_IMPLEMENTATION_UPDATER(base_class_name, class_name, alias) \
  class ImplementationUpdater_##base_class_name##class_name {         \
   public:                                                            \
    ImplementationUpdater_##base_class_name##class_name() {           \
      TC_IMPLEMENTATION_HOLDER_NAME(base_class_name)::get_instance()  \
          ->update<class_name>(alias);                                \
    }                                                                 \
  } ImplementationUpdater_##base_class_name##class_name##instance;

TC_NAMESPACE_END
