#pragma once

#include "config.h"
#include "asset_manager.h"
#include <cstring>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <iostream>

TC_NAMESPACE_BEGIN

template<typename T>
std::shared_ptr<T> create_instance(const std::string &alias);

template<typename T>
std::shared_ptr<T> create_initialized_instance(const std::string &alias, const Config &config);

#define TC_IMPLEMENTATION_HOLDER_NAME(class_name) ImplementationHolder##class_name

#define TC_INTERFACE(T) \
class ImplementationHolder##T { \
public: \
    ImplementationHolder##T(const std::string &name) { \
        this->name = name;\
        /*std::cout << "Interface [" << name << "] loaded." << std::endl;*/ \
    }\
    typedef std::function<std::shared_ptr<T>()> FactoryMethod; \
    std::string name; \
    std::map<std::string, FactoryMethod> implementation_factories; \
    template<typename G> \
    void register_implementation(const std::string &alias) { \
        /*std::cout << "Registering [" << alias << "] => [" << name << "]." << std::endl;*/ \
        implementation_factories.insert(std::make_pair(alias, [&]() { \
            return std::make_shared<G>(); \
        })); \
    } \
    std::shared_ptr<T> create(const std::string &alias) { \
        auto factory = implementation_factories.find(alias); \
        assert_info(factory != implementation_factories.end(), \
    "Implementation [" + name + "::" + alias + "] not found!"); \
return (factory->second)(); \
    } \
    static ImplementationHolder##T* get_instance(); \
};

#define TC_INTERFACE_DEF(class_name, base_alias) \
    TC_IMPLEMENTATION_HOLDER_NAME(class_name) *TC_IMPLEMENTATION_HOLDER_NAME(class_name)::get_instance() { \
        static TC_IMPLEMENTATION_HOLDER_NAME(class_name) instance = TC_IMPLEMENTATION_HOLDER_NAME(class_name)(base_alias); \
        return &instance;\
    }\
    template<> std::shared_ptr<class_name> create_instance(const std::string &alias) { \
        return TC_IMPLEMENTATION_HOLDER_NAME(class_name)::get_instance()->create(alias); \
    }\
    template<> std::shared_ptr<class_name> create_initialized_instance(const std::string &alias, const Config &config) { \
        auto instance = create_instance<class_name>(alias);\
        instance->initialize(config);\
        return instance;\
    }

#define TC_IMPLEMENTATION(base_class_name, class_name, alias) \
    class ImplementationInjector_##base_class_name##class_name {\
        public:\
        ImplementationInjector_##base_class_name##class_name() {\
            TC_IMPLEMENTATION_HOLDER_NAME(base_class_name)::get_instance()->register_implementation<class_name>(alias);\
        }\
    } ImplementationInjector_##base_class_name##class_name##instance;

TC_NAMESPACE_END
