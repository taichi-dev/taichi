/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/config.h>
#include <taichi/common/asset_manager.h>
#include <cstring>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <iostream>

TC_NAMESPACE_BEGIN

template <typename T>
TC_EXPORT std::shared_ptr<T> create_instance(const std::string &alias);

template <typename T>
TC_EXPORT std::shared_ptr<T> create_instance(const std::string &alias,
                                             const Config &config);

template <typename T>
TC_EXPORT std::unique_ptr<T> create_instance_unique(const std::string &alias);

template <typename T>
TC_EXPORT std::unique_ptr<T> create_instance_unique(const std::string &alias,
                                                    const Config &config);
template <typename T>
TC_EXPORT std::unique_ptr<T> create_instance_unique_ctor(
    const std::string &alias,
    const Config &config);

template <typename T>
TC_EXPORT T *create_instance_raw(const std::string &alias);

template <typename T>
TC_EXPORT T *create_instance_raw(const std::string &alias,
                                 const Config &config);

template <typename T>
TC_EXPORT T *create_instance_placement(const std::string &alias, void *place);

template <typename T>
TC_EXPORT T *create_instance_placement(const std::string &alias,
                                       void *place,
                                       const Config &config);

template <typename T>
TC_EXPORT std::vector<std::string> get_implementation_names();

class Unit {
 public:
  Unit() {
  }

  virtual void initialize(const Config &config) {
  }

  virtual bool test() const {
    return true;
  }

  virtual std::string get_name() const {
    TC_NOT_IMPLEMENTED;
    return "";
  }

  virtual std::string general_action(const Config &config) {
    TC_NOT_IMPLEMENTED;
    return "";
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

  void register_registration_method(const std::string &name,
                                    const RegistrationMethod &method) {
    methods[name] = method;
  }

  void register_interface(const std::string &name,
                          ImplementationHolderBase *interface_) {
    interfaces[name] = interface_;
  }

  static InterfaceHolder *get_instance() {
    static InterfaceHolder holder;
    return &holder;
  }
};

#define TC_INTERFACE(T)                                                       \
  extern void *get_implementation_holder_instance_##T();                      \
  class TC_IMPLEMENTATION_HOLDER_NAME(T) : public ImplementationHolderBase {  \
   public:                                                                    \
    TC_IMPLEMENTATION_HOLDER_NAME(T)(const std::string &name) {               \
      this->name = name;                                                      \
    }                                                                         \
    using FactoryMethod = std::function<std::shared_ptr<T>()>;                \
    using FactoryUniqueMethod = std::function<std::unique_ptr<T>()>;          \
    using FactoryUniqueCtorMethod =                                           \
        std::function<std::unique_ptr<T>(const Dict &config)>;                \
    using FactoryRawMethod = std::function<T *()>;                            \
    using FactoryPlacementMethod = std::function<T *(void *)>;                \
    std::map<std::string, FactoryMethod> implementation_factories;            \
    std::map<std::string, FactoryUniqueMethod>                                \
        implementation_unique_factories;                                      \
    std::map<std::string, FactoryUniqueCtorMethod>                            \
        implementation_unique_ctor_factories;                                 \
    std::map<std::string, FactoryRawMethod> implementation_raw_factories;     \
    std::map<std::string, FactoryPlacementMethod>                             \
        implementation_placement_factories;                                   \
    std::vector<std::string> get_implementation_names() const override {      \
      std::vector<std::string> names;                                         \
      for (auto &kv : implementation_factories) {                             \
        names.push_back(kv.first);                                            \
      }                                                                       \
      return names;                                                           \
    }                                                                         \
    template <typename G>                                                     \
    void insert(const std::string &alias) {                                   \
      implementation_factories.insert(                                        \
          std::make_pair(alias, [&]() { return std::make_shared<G>(); }));    \
      implementation_unique_factories.insert(                                 \
          std::make_pair(alias, [&]() { return std::make_unique<G>(); }));    \
      implementation_raw_factories.insert(                                    \
          std::make_pair(alias, [&]() { return new G(); }));                  \
      implementation_placement_factories.insert(std::make_pair(               \
          alias, [&](void *place) { return new (place) G(); }));              \
    }                                                                         \
    template <typename G>                                                     \
    void insert_new(const std::string &alias) {                               \
      /*with ctor*/                                                           \
      implementation_factories.insert(                                        \
          std::make_pair(alias, [&]() { return std::make_shared<G>(); }));    \
      implementation_unique_factories.insert(                                 \
          std::make_pair(alias, [&]() { return std::make_unique<G>(); }));    \
      implementation_unique_ctor_factories.insert(std::make_pair(             \
          alias,                                                              \
          [&](const Dict &config) { return std::make_unique<G>(config); }));  \
      implementation_raw_factories.insert(                                    \
          std::make_pair(alias, [&]() { return new G(); }));                  \
      implementation_placement_factories.insert(std::make_pair(               \
          alias, [&](void *place) { return new (place) G(); }));              \
    }                                                                         \
    void insert(const std::string &alias, const FactoryMethod &f) {           \
      implementation_factories.insert(std::make_pair(alias, f));              \
    }                                                                         \
    bool has(const std::string &alias) const override {                       \
      return implementation_factories.find(alias) !=                          \
             implementation_factories.end();                                  \
    }                                                                         \
    void remove(const std::string &alias) override {                          \
      assert_info(has(alias),                                                 \
                  std::string("Implemetation ") + alias + " not found!");     \
      implementation_factories.erase(alias);                                  \
    }                                                                         \
    void update(const std::string &alias, const FactoryMethod &f) {           \
      if (has(alias)) {                                                       \
        remove(alias);                                                        \
      }                                                                       \
      insert(alias, f);                                                       \
    }                                                                         \
    template <typename G>                                                     \
    void update(const std::string &alias) {                                   \
      if (has(alias)) {                                                       \
        remove(alias);                                                        \
      }                                                                       \
      insert<G>(alias);                                                       \
    }                                                                         \
    std::shared_ptr<T> create(const std::string &alias) {                     \
      auto factory = implementation_factories.find(alias);                    \
      assert_info(factory != implementation_factories.end(),                  \
                  "Implementation [" + name + "::" + alias + "] not found!"); \
      return (factory->second)();                                             \
    }                                                                         \
    std::unique_ptr<T> create_unique(const std::string &alias) {              \
      auto factory = implementation_unique_factories.find(alias);             \
      assert_info(factory != implementation_unique_factories.end(),           \
                  "Implementation [" + name + "::" + alias + "] not found!"); \
      return (factory->second)();                                             \
    }                                                                         \
    std::unique_ptr<T> create_unique_ctor(const std::string &alias,           \
                                          const Dict &config) {               \
      auto factory = implementation_unique_ctor_factories.find(alias);        \
      assert_info(factory != implementation_unique_ctor_factories.end(),      \
                  "Implementation [" + name + "::" + alias + "] not found!"); \
      return (factory->second)(config);                                       \
    }                                                                         \
    T *create_raw(const std::string &alias) {                                 \
      auto factory = implementation_raw_factories.find(alias);                \
      assert_info(factory != implementation_raw_factories.end(),              \
                  "Implementation [" + name + "::" + alias + "] not found!"); \
      return (factory->second)();                                             \
    }                                                                         \
    T *create_placement(const std::string &alias, void *place) {              \
      auto factory = implementation_placement_factories.find(alias);          \
      assert_info(factory != implementation_placement_factories.end(),        \
                  "Implementation [" + name + "::" + alias + "] not found!"); \
      return (factory->second)(place);                                        \
    }                                                                         \
    static TC_IMPLEMENTATION_HOLDER_NAME(T) * get_instance() {                \
      return static_cast<TC_IMPLEMENTATION_HOLDER_NAME(T) *>(                 \
          get_implementation_holder_instance_##T());                          \
    }                                                                         \
  };                                                                          \
  extern TC_IMPLEMENTATION_HOLDER_NAME(T) * TC_IMPLEMENTATION_HOLDER_PTR(T);

#define TC_INTERFACE_DEF(class_name, base_alias)                              \
  template <>                                                                 \
  TC_EXPORT std::shared_ptr<class_name> create_instance(                      \
      const std::string &alias) {                                             \
    return TC_IMPLEMENTATION_HOLDER_NAME(class_name)::get_instance()->create( \
        alias);                                                               \
  }                                                                           \
  template <>                                                                 \
  TC_EXPORT std::shared_ptr<class_name> create_instance(                      \
      const std::string &alias, const Config &config) {                       \
    auto instance = create_instance<class_name>(alias);                       \
    instance->initialize(config);                                             \
    return instance;                                                          \
  }                                                                           \
  template <>                                                                 \
  TC_EXPORT std::unique_ptr<class_name> create_instance_unique(               \
      const std::string &alias) {                                             \
    return TC_IMPLEMENTATION_HOLDER_NAME(class_name)::get_instance()          \
        ->create_unique(alias);                                               \
  }                                                                           \
  template <>                                                                 \
  TC_EXPORT std::unique_ptr<class_name> create_instance_unique(               \
      const std::string &alias, const Config &config) {                       \
    auto instance = create_instance_unique<class_name>(alias);                \
    instance->initialize(config);                                             \
    return instance;                                                          \
  }                                                                           \
  template <>                                                                 \
  TC_EXPORT std::unique_ptr<class_name> create_instance_unique_ctor(          \
      const std::string &alias, const Dict &config) {                         \
    return TC_IMPLEMENTATION_HOLDER_NAME(class_name)::get_instance()          \
        ->create_unique_ctor(alias, config);                                  \
  }                                                                           \
  template <>                                                                 \
  TC_EXPORT class_name *create_instance_raw(const std::string &alias) {       \
    return TC_IMPLEMENTATION_HOLDER_NAME(class_name)::get_instance()          \
        ->create_raw(alias);                                                  \
  }                                                                           \
  template <>                                                                 \
  TC_EXPORT class_name *create_instance_placement(const std::string &alias,   \
                                                  void *place) {              \
    return TC_IMPLEMENTATION_HOLDER_NAME(class_name)::get_instance()          \
        ->create_placement(alias, place);                                     \
  }                                                                           \
  template <>                                                                 \
  TC_EXPORT class_name *create_instance_placement(                            \
      const std::string &alias, void *place, const Config &config) {          \
    auto instance = create_instance_placement<class_name>(alias, place);      \
    instance->initialize(config);                                             \
    return instance;                                                          \
  }                                                                           \
  template <>                                                                 \
  TC_EXPORT class_name *create_instance_raw(const std::string &alias,         \
                                            const Config &config) {           \
    auto instance = create_instance_raw<class_name>(alias);                   \
    instance->initialize(config);                                             \
    return instance;                                                          \
  }                                                                           \
  template <>                                                                 \
  std::vector<std::string> get_implementation_names<class_name>() {           \
    return TC_IMPLEMENTATION_HOLDER_NAME(class_name)::get_instance()          \
        ->get_implementation_names();                                         \
  }                                                                           \
  TC_IMPLEMENTATION_HOLDER_NAME(class_name) *                                 \
      TC_IMPLEMENTATION_HOLDER_PTR(class_name) = nullptr;                     \
  void *get_implementation_holder_instance_##class_name() {                   \
    if (!TC_IMPLEMENTATION_HOLDER_PTR(class_name)) {                          \
      TC_IMPLEMENTATION_HOLDER_PTR(class_name) =                              \
          new TC_IMPLEMENTATION_HOLDER_NAME(class_name)(base_alias);          \
    }                                                                         \
    return TC_IMPLEMENTATION_HOLDER_PTR(class_name);                          \
  }                                                                           \
  class InterfaceInjector_##class_name {                                      \
   public:                                                                    \
    InterfaceInjector_##class_name(const std::string &name) {                 \
      InterfaceHolder::get_instance()->register_registration_method(          \
          base_alias, [&](void *m) {                                          \
            ((pybind11::module *)m)                                           \
                ->def("create_" base_alias,                                   \
                      static_cast<std::shared_ptr<class_name> (*)(            \
                          const std::string &name)>(                          \
                          &create_instance<class_name>));                     \
            ((pybind11::module *)m)                                           \
                ->def("register_" base_alias,                                 \
                      &AssetManager::insert_asset<class_name>);               \
            ((pybind11::module *)m)                                           \
                ->def("create_initialized_" base_alias,                       \
                      static_cast<std::shared_ptr<class_name> (*)(            \
                          const std::string &name, const Config &config)>(    \
                          &create_instance<class_name>));                     \
          });                                                                 \
      InterfaceHolder::get_instance()->register_interface(                    \
          base_alias, (ImplementationHolderBase *)                            \
                          get_implementation_holder_instance_##class_name()); \
    }                                                                         \
  } ImplementationInjector_##base_class_name##class_name##instance(base_alias);

#define TC_IMPLEMENTATION(base_class_name, class_name, alias)        \
  class ImplementationInjector_##base_class_name##class_name {       \
   public:                                                           \
    ImplementationInjector_##base_class_name##class_name() {         \
      TC_IMPLEMENTATION_HOLDER_NAME(base_class_name)::get_instance() \
          ->insert<class_name>(alias);                               \
    }                                                                \
  } ImplementationInjector_##base_class_name##class_name##instance;

#define TC_IMPLEMENTATION_NEW(base_class_name, class_name)           \
  class ImplementationInjector_##base_class_name##class_name {       \
   public:                                                           \
    ImplementationInjector_##base_class_name##class_name() {         \
      TC_IMPLEMENTATION_HOLDER_NAME(base_class_name)::get_instance() \
          ->insert_new<class_name>(class_name::get_name_static());   \
      TC_P(class_name::get_name_static());                           \
    }                                                                \
  } ImplementationInjector_##base_class_name##class_name##instance;

#define TC_NAME(alias)                            \
  virtual std::string get_name() const override { \
    return get_name_static();                     \
  }                                               \
  static std::string get_name_static() {          \
    return alias;                                 \
  }

TC_NAMESPACE_END
