/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "interface.h"

TI_NAMESPACE_BEGIN

#define TI_IMPLEMENTATION_LOADER(base_class_name, class_name, alias)          \
  class ImplementationLoader_##base_class_name##class_name {                  \
   public:                                                                    \
    ImplementationLoader_##base_class_name##class_name() {                    \
      TI_IMPLEMENTATION_HOLDER_NAME(base_class_name)::get_instance()          \
          ->insert<class_name>(alias);                                        \
    }                                                                         \
    ~ImplementationLoader_##base_class_name##class_name() {                   \
      TI_IMPLEMENTATION_HOLDER_NAME(base_class_name)::get_instance()->remove( \
          alias);                                                             \
    }                                                                         \
  } ImplementationLoader_##base_class_name##class_name##instance;

#define TI_IMPLEMENTATION_UPDATER(base_class_name, class_name, alias) \
  class ImplementationUpdater_##base_class_name##class_name {         \
   public:                                                            \
    ImplementationUpdater_##base_class_name##class_name() {           \
      TI_IMPLEMENTATION_HOLDER_NAME(base_class_name)::get_instance()  \
          ->update<class_name>(alias);                                \
    }                                                                 \
  } ImplementationUpdater_##base_class_name##class_name##instance;

TI_NAMESPACE_END
