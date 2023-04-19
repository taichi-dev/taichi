#pragma once

#include "lib_tiny_ir.h"
#include "taichi/ir/type.h"

namespace taichi::lang {
namespace spirv {

class STD140LayoutContext : public tinyir::LayoutContext {};
class STD430LayoutContext : public tinyir::LayoutContext {};

class IntType : public tinyir::Type, public tinyir::MemRefElementTypeInterface {
 public:
  IntType(int num_bits, bool is_signed)
      : num_bits_(num_bits), is_signed_(is_signed) {
  }

  int num_bits() const {
    return num_bits_;
  }

  bool is_signed() const {
    return is_signed_;
  }

  size_t memory_size(tinyir::LayoutContext &ctx) const override {
    return tinyir::ceil_div(num_bits(), 8);
  }

  size_t memory_alignment_size(tinyir::LayoutContext &ctx) const override {
    return tinyir::ceil_div(num_bits(), 8);
  }

 private:
  bool is_equal(const Polymorphic &other) const override {
    const IntType &t = (const IntType &)other;
    return t.num_bits_ == num_bits_ && t.is_signed_ == is_signed_;
  }

  int num_bits_{0};
  bool is_signed_{false};
};

class FloatType : public tinyir::Type,
                  public tinyir::MemRefElementTypeInterface {
 public:
  explicit FloatType(int num_bits) : num_bits_(num_bits) {
  }

  int num_bits() const {
    return num_bits_;
  }

  size_t memory_size(tinyir::LayoutContext &ctx) const override {
    return tinyir::ceil_div(num_bits(), 8);
  }

  size_t memory_alignment_size(tinyir::LayoutContext &ctx) const override {
    return tinyir::ceil_div(num_bits(), 8);
  }

 private:
  int num_bits_{0};

  bool is_equal(const Polymorphic &other) const override {
    const FloatType &t = (const FloatType &)other;
    return t.num_bits_ == num_bits_;
  }
};

class PhysicalPointerType : public IntType,
                            public tinyir::PointerTypeInterface {
 public:
  explicit PhysicalPointerType(const tinyir::Type *pointed_type)
      : IntType(/*num_bits=*/64, /*is_signed=*/false),
        pointed_type_(pointed_type) {
  }

  const tinyir::Type *get_pointed_type() const override {
    return pointed_type_;
  }

 private:
  const tinyir::Type *pointed_type_;

  bool is_equal(const Polymorphic &other) const override {
    const PhysicalPointerType &pt = (const PhysicalPointerType &)other;
    return IntType::operator==((const IntType &)other) &&
           pointed_type_->equals(pt.pointed_type_);
  }
};

class StructType : public tinyir::Type,
                   public tinyir::AggregateTypeInterface,
                   public tinyir::MemRefAggregateTypeInterface {
 public:
  explicit StructType(std::vector<const tinyir::Type *> &elements)
      : elements_(elements) {
  }

  const tinyir::Type *nth_element_type(int n) const override {
    return elements_[n];
  }

  int get_num_elements() const override {
    return elements_.size();
  }

  size_t memory_size(tinyir::LayoutContext &ctx) const override;

  size_t memory_alignment_size(tinyir::LayoutContext &ctx) const override;

  size_t nth_element_offset(int n, tinyir::LayoutContext &ctx) const override;

 private:
  std::vector<const tinyir::Type *> elements_;

  bool is_equal(const Polymorphic &other) const override {
    const StructType &t = (const StructType &)other;
    if (t.get_num_elements() != get_num_elements()) {
      return false;
    }
    for (int i = 0; i < get_num_elements(); i++) {
      if (!elements_[i]->equals(t.elements_[i])) {
        return false;
      }
    }
    return true;
  }
};

class SmallVectorType : public tinyir::Type,
                        public tinyir::ShapedTypeInterface,
                        public tinyir::MemRefElementTypeInterface {
 public:
  SmallVectorType(const tinyir::Type *element_type, int num_elements);

  const tinyir::Type *element_type() const override {
    return element_type_;
  }

  bool is_constant_shape() const override {
    return true;
  }

  std::vector<size_t> get_constant_shape() const override {
    return {size_t(num_elements_)};
  }

  size_t memory_size(tinyir::LayoutContext &ctx) const override;

  size_t memory_alignment_size(tinyir::LayoutContext &ctx) const override;

 private:
  bool is_equal(const Polymorphic &other) const override {
    const SmallVectorType &t = (const SmallVectorType &)other;
    return num_elements_ == t.num_elements_ &&
           element_type_->equals(t.element_type_);
  }

  const tinyir::Type *element_type_{nullptr};
  int num_elements_{0};
};

class ArrayType : public tinyir::Type,
                  public tinyir::ShapedTypeInterface,
                  public tinyir::MemRefAggregateTypeInterface {
 public:
  ArrayType(const tinyir::Type *element_type, size_t size)
      : element_type_(element_type), size_(size) {
  }

  const tinyir::Type *element_type() const override {
    return element_type_;
  }

  bool is_constant_shape() const override {
    return true;
  }

  std::vector<size_t> get_constant_shape() const override {
    return {size_};
  }

  size_t memory_size(tinyir::LayoutContext &ctx) const override;

  size_t memory_alignment_size(tinyir::LayoutContext &ctx) const override;

  size_t nth_element_offset(int n, tinyir::LayoutContext &ctx) const override;

 private:
  bool is_equal(const Polymorphic &other) const override {
    const ArrayType &t = (const ArrayType &)other;
    return size_ == t.size_ && element_type_->equals(t.element_type_);
  }

  const tinyir::Type *element_type_{nullptr};
  size_t size_{0};
};

bool bitcast_possible(tinyir::Type *a, tinyir::Type *b, bool _inverted = false);

class TypeVisitor : public tinyir::Visitor {
 public:
  void visit_type(const tinyir::Type *type) override;

  virtual void visit_int_type(const IntType *type) {
  }

  virtual void visit_float_type(const FloatType *type) {
  }

  virtual void visit_physical_pointer_type(const PhysicalPointerType *type) {
  }

  virtual void visit_struct_type(const StructType *type) {
  }

  virtual void visit_small_vector_type(const SmallVectorType *type) {
  }

  virtual void visit_array_type(const ArrayType *type) {
  }
};

const tinyir::Type *translate_ti_primitive(tinyir::Block &ir_module,
                                           const DataType t);

const tinyir::Type *translate_ti_type(tinyir::Block &ir_module,
                                      const DataType t,
                                      bool has_buffer_ptr);

std::string ir_print_types(const tinyir::Block *block);

std::unique_ptr<tinyir::Block> ir_reduce_types(
    tinyir::Block *blk,
    std::unordered_map<const tinyir::Type *, const tinyir::Type *> &old2new);

class IRBuilder;

std::unordered_map<const tinyir::Node *, uint32_t> ir_translate_to_spirv(
    const tinyir::Block *blk,
    tinyir::LayoutContext &layout_ctx,
    IRBuilder *spir_builder);

}  // namespace spirv
}  // namespace taichi::lang
