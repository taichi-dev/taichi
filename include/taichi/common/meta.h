/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/config.h>
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

class Unit {
public:
    Unit() {}

    virtual void initialize(const Config &config) {

    }

    virtual bool test() const {
        return true;
    }

    virtual std::string get_name() {
        return "unit";
    }
};

#define TC_IMPLEMENTATION_HOLDER_NAME(T) ImplementationHolder_##T
#define TC_IMPLEMENTATION_HOLDER_PTR(T) instance_ImplementationHolder_##T

#define TC_INTERFACE(T) \
extern void *get_implementation_holder_instance_##T();\
class TC_IMPLEMENTATION_HOLDER_NAME(T) { \
public: \
    TC_IMPLEMENTATION_HOLDER_NAME(T)(const std::string &name) { \
        this->name = name;\
    }\
    typedef std::function<std::shared_ptr<T>()> FactoryMethod; \
    std::string name; \
    std::map<std::string, FactoryMethod> implementation_factories; \
    template<typename G> \
    void insert(const std::string &alias) { \
        implementation_factories.insert(std::make_pair(alias, [&]() { \
            return std::make_shared<G>(); \
        })); \
    } \
    void insert(const std::string &alias, const FactoryMethod &f) { \
        implementation_factories.insert(std::make_pair(alias, f)); \
    } \
    bool has(const std::string &alias) const { \
        return implementation_factories.find(alias) != implementation_factories.end(); \
    } \
    void remove(const std::string &alias) { \
        assert_info(has(alias), std::string("Implemetation ") + alias + " not found!"); \
        implementation_factories.erase(alias); \
    } \
    void update(const std::string &alias, const FactoryMethod &f) { \
        if (has(alias)) { \
            remove(alias); \
        } \
        insert(alias, f); \
    } \
    template<typename G> \
    void update(const std::string &alias) { \
        if (has(alias)) { \
            remove(alias); \
        } \
        insert<G>(alias); \
    } \
    std::shared_ptr<T> create(const std::string &alias) { \
        auto factory = implementation_factories.find(alias); \
        assert_info(factory != implementation_factories.end(), \
    "Implementation [" + name + "::" + alias + "] not found!"); \
return (factory->second)(); \
    } \
    static TC_IMPLEMENTATION_HOLDER_NAME(T)* get_instance() { \
        return static_cast<TC_IMPLEMENTATION_HOLDER_NAME(T) *>(get_implementation_holder_instance_##T()); \
    } \
}; \
extern TC_IMPLEMENTATION_HOLDER_NAME(T) *TC_IMPLEMENTATION_HOLDER_PTR(T);

#define TC_INTERFACE_DEF(class_name, base_alias) \
    TC_IMPLEMENTATION_HOLDER_NAME(class_name) *TC_IMPLEMENTATION_HOLDER_PTR(class_name) = nullptr; \
    void *get_implementation_holder_instance_##class_name() { \
        if (!TC_IMPLEMENTATION_HOLDER_PTR(class_name)) { \
            TC_IMPLEMENTATION_HOLDER_PTR(class_name) = new TC_IMPLEMENTATION_HOLDER_NAME(class_name)(base_alias); \
        } \
        return TC_IMPLEMENTATION_HOLDER_PTR(class_name); \
    } \
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
            TC_IMPLEMENTATION_HOLDER_NAME(base_class_name)::get_instance()->insert<class_name>(alias);\
        }\
    } ImplementationInjector_##base_class_name##class_name##instance;

TC_NAMESPACE_END
