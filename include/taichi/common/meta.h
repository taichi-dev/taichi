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
std::shared_ptr<T> create_instance(const std::string &alias, const Config &config);

template<typename T>
std::vector<std::string> get_implementation_names();

class Unit {
public:
    Unit() {}

    virtual void initialize(const Config &config) {}

    virtual bool test() const {
        return true;
    }

    virtual std::string get_name() const {
        return "unit";
    }
};

#define TC_IMPLEMENTATION_HOLDER_NAME(T) ImplementationHolder_##T
#define TC_IMPLEMENTATION_HOLDER_PTR(T) instance_ImplementationHolder_##T

class ImplementationHolderBase {
public:
    std::string name;
    virtual bool has(const std::string &alias) const = 0;
    virtual void remove(const std::string &alias) = 0;
    virtual std::vector<std::string> get_implementation_names() const = 0;
};

class InterfaceHolder {
public:
    typedef std::function<void(void *)> RegistrationMethod;
    std::map<std::string, RegistrationMethod> methods;
    std::map<std::string, ImplementationHolderBase *> interfaces;
    void register_registration_method(const std::string &name, const RegistrationMethod &method) {
        methods[name] = method;
    }
    void register_interface(const std::string &name, ImplementationHolderBase *interface) {
        interfaces[name] = interface;
    }
    static InterfaceHolder* get_instance() {
        static InterfaceHolder holder;
        return &holder;
    }
};

#define TC_INTERFACE(T) \
extern void *get_implementation_holder_instance_##T();\
class TC_IMPLEMENTATION_HOLDER_NAME(T) : public ImplementationHolderBase { \
public: \
    TC_IMPLEMENTATION_HOLDER_NAME(T)(const std::string &name) { \
        this->name = name;\
    }\
    typedef std::function<std::shared_ptr<T>()> FactoryMethod; \
    std::map<std::string, FactoryMethod> implementation_factories; \
    std::vector<std::string> get_implementation_names() const override { \
        std::vector<std::string> names; \
        for (auto &kv : implementation_factories) { \
            names.push_back(kv.first); \
        } \
        return names; \
    } \
    template<typename G> \
    void insert(const std::string &alias) { \
        implementation_factories.insert(std::make_pair(alias, [&]() { \
            return std::make_shared<G>(); \
        })); \
    } \
    void insert(const std::string &alias, const FactoryMethod &f) { \
        implementation_factories.insert(std::make_pair(alias, f)); \
    } \
    bool has(const std::string &alias) const override { \
        return implementation_factories.find(alias) != implementation_factories.end(); \
    } \
    void remove(const std::string &alias) override { \
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
    template<> std::shared_ptr<class_name> create_instance(const std::string &alias) { \
        return TC_IMPLEMENTATION_HOLDER_NAME(class_name)::get_instance()->create(alias); \
    } \
    template<> std::shared_ptr<class_name> create_instance(const std::string &alias, const Config &config) { \
        auto instance = create_instance<class_name>(alias); \
        instance->initialize(config); \
        return instance; \
    } \
    template<> std::vector<std::string> get_implementation_names<class_name>() { \
        return TC_IMPLEMENTATION_HOLDER_NAME(class_name)::get_instance()->get_implementation_names(); \
    } \
    TC_IMPLEMENTATION_HOLDER_NAME(class_name) *TC_IMPLEMENTATION_HOLDER_PTR(class_name) = nullptr; \
    void *get_implementation_holder_instance_##class_name() { \
        if (!TC_IMPLEMENTATION_HOLDER_PTR(class_name)) { \
            TC_IMPLEMENTATION_HOLDER_PTR(class_name) = new TC_IMPLEMENTATION_HOLDER_NAME(class_name)(base_alias); \
        } \
        return TC_IMPLEMENTATION_HOLDER_PTR(class_name); \
    } \
    class InterfaceInjector_##class_name {\
        public:\
        InterfaceInjector_##class_name(const std::string &name) {\
            InterfaceHolder::get_instance()->register_registration_method(base_alias, [&](void *m) {\
                ((pybind11::module *)m)->def("create_" base_alias, \
                    static_cast<std::shared_ptr<class_name>(*)(const std::string &name)>(&create_instance<class_name>)); \
                ((pybind11::module *)m)->def("create_initialized_" base_alias, \
                    static_cast<std::shared_ptr<class_name>(*)(const std::string &name, \
                    const Config &config)>(&create_instance<class_name>)); \
            });\
            InterfaceHolder::get_instance()->register_interface(base_alias, \
                (ImplementationHolderBase *)get_implementation_holder_instance_##class_name());\
        }\
    } ImplementationInjector_##base_class_name##class_name##instance(base_alias);

#define TC_IMPLEMENTATION(base_class_name, class_name, alias) \
    class ImplementationInjector_##base_class_name##class_name {\
        public:\
        ImplementationInjector_##base_class_name##class_name() {\
            TC_IMPLEMENTATION_HOLDER_NAME(base_class_name)::get_instance()->insert<class_name>(alias);\
        }\
    } ImplementationInjector_##base_class_name##class_name##instance;

TC_NAMESPACE_END
