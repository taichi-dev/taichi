#pragma once

#include <taichi/common/meta.h>

TC_NAMESPACE_BEGIN

#define TC_IMPLEMENTATION_LOADER(base_class_name, class_name, alias) \
class ImplementationLoader_##base_class_name##class_name {\
public:\
    ImplementationLoader_##base_class_name##class_name() {\
        TC_IMPLEMENTATION_HOLDER_NAME(base_class_name)::get_instance()->insert<class_name>(alias);\
    }\
    ~ImplementationLoader_##base_class_name##class_name() {\
        TC_IMPLEMENTATION_HOLDER_NAME(base_class_name)::get_instance()->remove(alias);\
    }\
} ImplementationLoader_##base_class_name##class_name##instance;

TC_NAMESPACE_END

