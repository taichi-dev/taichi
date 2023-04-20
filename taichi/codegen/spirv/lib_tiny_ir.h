#pragma once

#include "taichi/common/core.h"

#include <vector>
#include <memory>

namespace taichi {
namespace tinyir {

template <typename T>
T ceil_div(T v, T div) {
  return (v / div) + (v % div ? 1 : 0);
}

// Forward decl
class Polymorphic;
class Node;
class Type;
class LayoutContext;
class MemRefElementTypeInterface;
class MemRefAggregateTypeInterface;
class ShapedTypeInterface;
class AggregateTypeInterface;
class PointerTypeInterface;
class Block;
class Visitor;

class Polymorphic {
 public:
  virtual ~Polymorphic() {
  }

  template <typename T>
  bool is() const {
    return dynamic_cast<const T *>(this) != nullptr;
  }

  template <typename T>
  T *as() {
    return static_cast<T *>(this);
  }

  template <typename T>
  const T *as() const {
    return static_cast<const T *>(this);
  }

  template <typename T>
  T *cast() {
    return dynamic_cast<T *>(this);
  }

  template <typename T>
  const T *cast() const {
    return dynamic_cast<const T *>(this);
  }

  bool operator==(const Polymorphic &other) const {
    return typeid(*this) == typeid(other) && is_equal(other);
  }

  bool equals(const Polymorphic *other) const {
    return (*this) == (*other);
  }

 private:
  virtual bool is_equal(const Polymorphic &other) const = 0;
};

class Node : public Polymorphic {
 public:
  using NodeRefs = const std::vector<const Node *>;

  Node() {
  }

  ~Node() override {
  }

  const std::string &debug_name() const {
    return debug_name_;
  }

  void set_debug_name(const std::string &s) {
    debug_name_ = s;
  }

  virtual NodeRefs incoming() const {
    return {};
  }

  virtual NodeRefs outgoing() const {
    return {};
  }

  virtual bool is_leaf() const {
    return false;
  }

  virtual bool is_tree_node() const {
    return false;
  }

 private:
  bool is_equal(const Polymorphic &other) const override {
    return false;
  }

  std::string debug_name_;
};

class Type : public Node {
 public:
  Type() {
  }

 private:
  bool is_equal(const Polymorphic &other) const override {
    return false;
  }
};

// The default LayoutContext is the standard C layout
class LayoutContext : public Polymorphic {
 private:
  std::unordered_map<const MemRefElementTypeInterface *, size_t> size_cache_;
  std::unordered_map<const MemRefElementTypeInterface *, size_t>
      alignment_cache_;
  std::unordered_map<const MemRefAggregateTypeInterface *, std::vector<size_t>>
      elem_offset_cache_;

 public:
  void register_size(const MemRefElementTypeInterface *t, size_t size) {
    TI_ASSERT(size != 0);
    size_cache_[t] = size;
  }

  void register_alignment(const MemRefElementTypeInterface *t, size_t size) {
    TI_ASSERT(size != 0);
    alignment_cache_[t] = size;
  }

  void register_aggregate(const MemRefAggregateTypeInterface *t, int num_elem) {
    elem_offset_cache_[t] = {};
    elem_offset_cache_[t].resize(num_elem, 0);
  }

  void register_elem_offset(const MemRefAggregateTypeInterface *t,
                            int n,
                            size_t offset) {
    TI_ASSERT(elem_offset_cache_.find(t) != elem_offset_cache_.end());
    elem_offset_cache_[t][n] = offset;
  }

  // Size or alignment can not be zero
  size_t query_size(const MemRefElementTypeInterface *t) {
    if (size_cache_.find(t) != size_cache_.end()) {
      return size_cache_[t];
    } else {
      return 0;
    }
  }

  size_t query_alignment(const MemRefElementTypeInterface *t) {
    if (alignment_cache_.find(t) != alignment_cache_.end()) {
      return alignment_cache_[t];
    } else {
      return 0;
    }
  }

  size_t query_elem_offset(const MemRefAggregateTypeInterface *t, int n) {
    if (elem_offset_cache_.find(t) != elem_offset_cache_.end()) {
      return elem_offset_cache_[t][n];
    } else {
      return 0;
    }
  }

 private:
  bool is_equal(const Polymorphic &other) const override {
    // This is only called when `other` has the same typeid
    return true;
  }
};

class MemRefElementTypeInterface {
 public:
  virtual size_t memory_size(LayoutContext &ctx) const = 0;
  virtual size_t memory_alignment_size(LayoutContext &ctx) const = 0;
};

class MemRefAggregateTypeInterface : public MemRefElementTypeInterface {
 public:
  virtual size_t nth_element_offset(int n, LayoutContext &ctx) const = 0;
};

class AggregateTypeInterface {
 public:
  virtual const Type *nth_element_type(int n) const = 0;
  virtual int get_num_elements() const = 0;
};

class ShapedTypeInterface {
 public:
  virtual const Type *element_type() const = 0;
  virtual bool is_constant_shape() const = 0;
  virtual std::vector<size_t> get_constant_shape() const = 0;
};

class PointerTypeInterface {
 public:
  virtual const Type *get_pointed_type() const = 0;
};

class Block {
 public:
  template <typename T, class... E>
  T *emplace_back(E... args) {
    nodes_.push_back(std::make_unique<T>(args...));
    return static_cast<T *>(nodes_.back().get());
  }

  template <typename T>
  T *push_back(std::unique_ptr<T> &&val) {
    T *ptr = val.get();
    nodes_.push_back(std::move(val));
    return ptr;
  }

  const std::vector<std::unique_ptr<Node>> &nodes() const {
    return nodes_;
  }

 private:
  std::vector<std::unique_ptr<Node>> nodes_;
};

class Visitor {
 public:
  virtual ~Visitor() {
  }

  virtual void visit(const Node *node) {
    if (node->is<Type>()) {
      visit_type(node->as<Type>());
    }
  }

  virtual void visit_type(const Type *type) {
  }

  virtual void visit(const Block *block) {
    for (auto &n : block->nodes()) {
      visit(n.get());
    }
  }
};

}  // namespace tinyir
}  // namespace taichi
