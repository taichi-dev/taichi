#include "spirv_types.h"
#include "spirv_ir_builder.h"

namespace taichi::lang {
namespace spirv {

size_t StructType::memory_size(tinyir::LayoutContext &ctx) const {
  if (size_t s = ctx.query_size(this)) {
    return s;
  }

  ctx.register_aggregate(this, elements_.size());

  size_t size_head = 0;
  int n = 0;
  for (const Type *elem : elements_) {
    TI_ASSERT(elem->is<tinyir::MemRefElementTypeInterface>());
    const MemRefElementTypeInterface *mem_ref_type =
        elem->cast<tinyir::MemRefElementTypeInterface>();
    size_t elem_size = mem_ref_type->memory_size(ctx);
    size_t elem_align = mem_ref_type->memory_alignment_size(ctx);
    // First align the head ptr, then add the size
    size_head = tinyir::ceil_div(size_head, elem_align) * elem_align;
    ctx.register_elem_offset(this, n, size_head);
    size_head += elem_size;
    n++;
  }

  if (ctx.is<STD140LayoutContext>()) {
    // With STD140 layout, the next member is rounded up to the alignment size.
    // Thus we should simply size up the struct to the alignment.
    size_t self_alignment = this->memory_alignment_size(ctx);
    size_head = tinyir::ceil_div(size_head, self_alignment) * self_alignment;
  }

  ctx.register_size(this, size_head);
  return size_head;
}

size_t StructType::memory_alignment_size(tinyir::LayoutContext &ctx) const {
  if (size_t s = ctx.query_alignment(this)) {
    return s;
  }

  size_t max_align = 0;
  for (const Type *elem : elements_) {
    TI_ASSERT(elem->is<tinyir::MemRefElementTypeInterface>());
    max_align = std::max(
        max_align,
        elem->cast<MemRefElementTypeInterface>()->memory_alignment_size(ctx));
  }

  if (ctx.is<STD140LayoutContext>()) {
    // With STD140 layout, struct alignment is rounded up to `sizeof(vec4)`
    constexpr size_t vec4_size = sizeof(float) * 4;
    max_align = tinyir::ceil_div(max_align, vec4_size) * vec4_size;
  }

  ctx.register_alignment(this, max_align);
  return max_align;
}

size_t StructType::nth_element_offset(int n, tinyir::LayoutContext &ctx) const {
  this->memory_size(ctx);

  return ctx.query_elem_offset(this, n);
}

SmallVectorType::SmallVectorType(const Type *element_type, int num_elements)
    : element_type_(element_type), num_elements_(num_elements) {
  TI_ASSERT(num_elements > 1 && num_elements_ <= 4);
}

size_t SmallVectorType::memory_size(tinyir::LayoutContext &ctx) const {
  if (size_t s = ctx.query_size(this)) {
    return s;
  }

  size_t size =
      element_type_->cast<tinyir::MemRefElementTypeInterface>()->memory_size(
          ctx) *
      num_elements_;

  ctx.register_size(this, size);
  return size;
}

size_t SmallVectorType::memory_alignment_size(
    tinyir::LayoutContext &ctx) const {
  if (size_t s = ctx.query_alignment(this)) {
    return s;
  }

  size_t align =
      element_type_->cast<tinyir::MemRefElementTypeInterface>()->memory_size(
          ctx);

  if (ctx.is<STD430LayoutContext>() || ctx.is<STD140LayoutContext>()) {
    // For STD140 / STD430, small vectors are Power-of-Two aligned
    // In C or "Scalar block layout", blocks are aligned to its component
    // alignment
    if (num_elements_ == 2) {
      align *= 2;
    } else {
      align *= 4;
    }
  }

  ctx.register_alignment(this, align);
  return align;
}

size_t ArrayType::memory_size(tinyir::LayoutContext &ctx) const {
  if (size_t s = ctx.query_size(this)) {
    return s;
  }

  size_t elem_align = element_type_->cast<tinyir::MemRefElementTypeInterface>()
                          ->memory_alignment_size(ctx);

  if (ctx.is<STD140LayoutContext>()) {
    // For STD140, arrays element stride equals the base alignment of the array
    // itself
    elem_align = this->memory_alignment_size(ctx);
  }
  size_t size = elem_align * size_;

  ctx.register_size(this, size);
  return size;
}

size_t ArrayType::memory_alignment_size(tinyir::LayoutContext &ctx) const {
  if (size_t s = ctx.query_alignment(this)) {
    return s;
  }

  size_t elem_align = element_type_->cast<tinyir::MemRefElementTypeInterface>()
                          ->memory_alignment_size(ctx);

  if (ctx.is<STD140LayoutContext>()) {
    // With STD140 layout, array alignment is rounded up to `sizeof(vec4)`
    constexpr size_t vec4_size = sizeof(float) * 4;
    elem_align = tinyir::ceil_div(elem_align, vec4_size) * vec4_size;
  }

  ctx.register_alignment(this, elem_align);
  return elem_align;
}

size_t ArrayType::nth_element_offset(int n, tinyir::LayoutContext &ctx) const {
  size_t elem_align = this->memory_alignment_size(ctx);

  return elem_align * n;
}

bool bitcast_possible(tinyir::Type *a, tinyir::Type *b, bool _inverted) {
  if (a->is<IntType>() && b->is<IntType>()) {
    return a->as<IntType>()->num_bits() == b->as<IntType>()->num_bits();
  } else if (a->is<FloatType>() && b->is<IntType>()) {
    return a->as<FloatType>()->num_bits() == b->as<IntType>()->num_bits();
  } else if (!_inverted) {
    return bitcast_possible(b, a, true);
  }
  return false;
}

const tinyir::Type *translate_ti_primitive(tinyir::Block &ir_module,
                                           const DataType t) {
  if (t->is<PrimitiveType>()) {
    if (t == PrimitiveType::i8) {
      return ir_module.emplace_back<IntType>(/*num_bits=*/8,
                                             /*is_signed=*/true);
    } else if (t == PrimitiveType::i16) {
      return ir_module.emplace_back<IntType>(/*num_bits=*/16,
                                             /*is_signed=*/true);
    } else if (t == PrimitiveType::i32) {
      return ir_module.emplace_back<IntType>(/*num_bits=*/32,
                                             /*is_signed=*/true);
    } else if (t == PrimitiveType::i64) {
      return ir_module.emplace_back<IntType>(/*num_bits=*/64,
                                             /*is_signed=*/true);
    } else if (t == PrimitiveType::u1) {
      // Spir-v has no full support for boolean types, using boolean types in
      // backend may cause issues. These issues arise when we use boolean as
      // return type, argument type and inner dtype of compount types. Since
      // boolean types has the same width with int32 in GLSL, we use int32
      // instead.
      return ir_module.emplace_back<IntType>(/*num_bits=*/32,
                                             /*is_signed=*/true);
    } else if (t == PrimitiveType::u8) {
      return ir_module.emplace_back<IntType>(/*num_bits=*/8,
                                             /*is_signed=*/false);
    } else if (t == PrimitiveType::u16) {
      return ir_module.emplace_back<IntType>(/*num_bits=*/16,
                                             /*is_signed=*/false);
    } else if (t == PrimitiveType::u32) {
      return ir_module.emplace_back<IntType>(/*num_bits=*/32,
                                             /*is_signed=*/false);
    } else if (t == PrimitiveType::u64) {
      return ir_module.emplace_back<IntType>(/*num_bits=*/64,
                                             /*is_signed=*/false);
    } else if (t == PrimitiveType::f16) {
      return ir_module.emplace_back<FloatType>(/*num_bits=*/16);
    } else if (t == PrimitiveType::f32) {
      return ir_module.emplace_back<FloatType>(/*num_bits=*/32);
    } else if (t == PrimitiveType::f64) {
      return ir_module.emplace_back<FloatType>(/*num_bits=*/64);
    } else {
      TI_NOT_IMPLEMENTED;
    }
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

void TypeVisitor::visit_type(const tinyir::Type *type) {
  if (type->is<PhysicalPointerType>()) {
    visit_physical_pointer_type(type->as<PhysicalPointerType>());
  } else if (type->is<SmallVectorType>()) {
    visit_small_vector_type(type->as<SmallVectorType>());
  } else if (type->is<ArrayType>()) {
    visit_array_type(type->as<ArrayType>());
  } else if (type->is<StructType>()) {
    visit_struct_type(type->as<StructType>());
  } else if (type->is<IntType>()) {
    visit_int_type(type->as<IntType>());
  } else if (type->is<FloatType>()) {
    visit_float_type(type->as<FloatType>());
  }
}

class TypePrinter : public TypeVisitor {
 private:
  std::string result_;
  STD140LayoutContext layout_ctx_;

  uint32_t head_{0};
  std::unordered_map<const tinyir::Type *, uint32_t> idmap_;

  uint32_t get_id(const tinyir::Type *type) {
    if (idmap_.find(type) == idmap_.end()) {
      uint32_t id = head_++;
      idmap_[type] = id;
      return id;
    } else {
      return idmap_[type];
    }
  }

 public:
  void visit_int_type(const IntType *type) override {
    result_ += fmt::format("T{} = {}int{}_t\n", get_id(type),
                           type->is_signed() ? "" : "u", type->num_bits());
  }

  void visit_float_type(const FloatType *type) override {
    result_ += fmt::format("T{} = float{}_t\n", get_id(type), type->num_bits());
  }

  void visit_physical_pointer_type(const PhysicalPointerType *type) override {
    result_ += fmt::format("T{} = T{} *\n", get_id(type),
                           get_id(type->get_pointed_type()));
  }

  void visit_struct_type(const StructType *type) override {
    result_ += fmt::format("T{} = struct {{", get_id(type));
    for (int i = 0; i < type->get_num_elements(); i++) {
      result_ += fmt::format("T{}, ", get_id(type->nth_element_type(i)));
    }
    result_ += "}}\n";
  }

  void visit_small_vector_type(const SmallVectorType *type) override {
    result_ += fmt::format("T{} = small_vector<T{}, {}>\n", get_id(type),
                           get_id(type->element_type()),
                           type->get_constant_shape()[0]);
  }

  void visit_array_type(const ArrayType *type) override {
    result_ += fmt::format("T{} = array<T{}, {}>\n", get_id(type),
                           get_id(type->element_type()),
                           type->get_constant_shape()[0]);
  }

  static std::string print_types(const tinyir::Block *block) {
    TypePrinter p;
    p.visit(block);
    return p.result_;
  }
};

std::string ir_print_types(const tinyir::Block *block) {
  return TypePrinter::print_types(block);
}

class TypeReducer : public TypeVisitor {
 public:
  std::unique_ptr<tinyir::Block> copy{nullptr};
  std::unordered_map<const tinyir::Type *, const tinyir::Type *> &oldptr2newptr;

  explicit TypeReducer(
      std::unordered_map<const tinyir::Type *, const tinyir::Type *> &old2new)
      : oldptr2newptr(old2new) {
    copy = std::make_unique<tinyir::Block>();
    old2new.clear();
  }

  const tinyir::Type *check_type(const tinyir::Type *type) {
    if (oldptr2newptr.find(type) != oldptr2newptr.end()) {
      return oldptr2newptr[type];
    }
    for (const auto &t : copy->nodes()) {
      if (t->equals(type)) {
        oldptr2newptr[type] = (const tinyir::Type *)t.get();
        return (const tinyir::Type *)t.get();
      }
    }
    return nullptr;
  }

  void visit_int_type(const IntType *type) override {
    if (!check_type(type)) {
      oldptr2newptr[type] = copy->emplace_back<IntType>(*type);
    }
  }

  void visit_float_type(const FloatType *type) override {
    if (!check_type(type)) {
      oldptr2newptr[type] = copy->emplace_back<FloatType>(*type);
    }
  }

  void visit_physical_pointer_type(const PhysicalPointerType *type) override {
    if (!check_type(type)) {
      const tinyir::Type *pointed = check_type(type->get_pointed_type());
      TI_ASSERT(pointed);
      oldptr2newptr[type] = copy->emplace_back<PhysicalPointerType>(pointed);
    }
  }

  void visit_struct_type(const StructType *type) override {
    if (!check_type(type)) {
      std::vector<const tinyir::Type *> elements;
      for (int i = 0; i < type->get_num_elements(); i++) {
        const tinyir::Type *elm = check_type(type->nth_element_type(i));
        TI_ASSERT(elm);
        elements.push_back(elm);
      }
      oldptr2newptr[type] = copy->emplace_back<StructType>(elements);
    }
  }

  void visit_small_vector_type(const SmallVectorType *type) override {
    if (!check_type(type)) {
      const tinyir::Type *element = check_type(type->element_type());
      TI_ASSERT(element);
      oldptr2newptr[type] = copy->emplace_back<SmallVectorType>(
          element, type->get_constant_shape()[0]);
    }
  }

  void visit_array_type(const ArrayType *type) override {
    if (!check_type(type)) {
      const tinyir::Type *element = check_type(type->element_type());
      TI_ASSERT(element);
      oldptr2newptr[type] =
          copy->emplace_back<ArrayType>(element, type->get_constant_shape()[0]);
    }
  }
};

std::unique_ptr<tinyir::Block> ir_reduce_types(
    tinyir::Block *blk,
    std::unordered_map<const tinyir::Type *, const tinyir::Type *> &old2new) {
  TypeReducer reducer(old2new);
  reducer.visit(blk);
  return std::move(reducer.copy);
}

class Translate2Spirv : public TypeVisitor {
 private:
  IRBuilder *spir_builder_{nullptr};
  tinyir::LayoutContext &layout_context_;

 public:
  std::unordered_map<const tinyir::Node *, uint32_t> ir_node_2_spv_value;

  Translate2Spirv(IRBuilder *spir_builder,
                  tinyir::LayoutContext &layout_context)
      : spir_builder_(spir_builder), layout_context_(layout_context) {
  }

  void visit_int_type(const IntType *type) override {
    SType vt;
    if (type->is_signed()) {
      if (type->num_bits() == 8) {
        vt = spir_builder_->i8_type();
      } else if (type->num_bits() == 16) {
        vt = spir_builder_->i16_type();
      } else if (type->num_bits() == 32) {
        vt = spir_builder_->i32_type();
      } else if (type->num_bits() == 64) {
        vt = spir_builder_->i64_type();
      }
    } else {
      if (type->num_bits() == 1) {
        vt = spir_builder_->bool_type();
      } else if (type->num_bits() == 8) {
        vt = spir_builder_->u8_type();
      } else if (type->num_bits() == 16) {
        vt = spir_builder_->u16_type();
      } else if (type->num_bits() == 32) {
        vt = spir_builder_->u32_type();
      } else if (type->num_bits() == 64) {
        vt = spir_builder_->u64_type();
      }
    }
    ir_node_2_spv_value[type] = vt.id;
  }

  void visit_float_type(const FloatType *type) override {
    SType vt;
    if (type->num_bits() == 16) {
      vt = spir_builder_->f16_type();
    } else if (type->num_bits() == 32) {
      vt = spir_builder_->f32_type();
    } else if (type->num_bits() == 64) {
      vt = spir_builder_->f64_type();
    }
    ir_node_2_spv_value[type] = vt.id;
  }

  void visit_physical_pointer_type(const PhysicalPointerType *type) override {
    SType vt = spir_builder_->get_null_type();
    spir_builder_->declare_global(
        spv::OpTypePointer, vt, spv::StorageClassPhysicalStorageBuffer,
        ir_node_2_spv_value[type->get_pointed_type()]);
    ir_node_2_spv_value[type] = vt.id;
  }

  void visit_struct_type(const StructType *type) override {
    std::vector<uint32_t> element_ids;
    for (int i = 0; i < type->get_num_elements(); i++) {
      element_ids.push_back(ir_node_2_spv_value[type->nth_element_type(i)]);
    }
    SType vt = spir_builder_->get_null_type();
    spir_builder_->declare_global(spv::OpTypeStruct, vt, element_ids);
    ir_node_2_spv_value[type] = vt.id;
    for (int i = 0; i < type->get_num_elements(); i++) {
      spir_builder_->decorate(spv::OpMemberDecorate, vt, i,
                              spv::DecorationOffset,
                              type->nth_element_offset(i, layout_context_));
    }
  }

  void visit_small_vector_type(const SmallVectorType *type) override {
    SType vt = spir_builder_->get_null_type();
    spir_builder_->declare_global(spv::OpTypeVector, vt,
                                  ir_node_2_spv_value[type->element_type()],
                                  type->get_constant_shape()[0]);
    ir_node_2_spv_value[type] = vt.id;
  }

  void visit_array_type(const ArrayType *type) override {
    SType vt = spir_builder_->get_null_type();
    spir_builder_->declare_global(
        spv::OpTypeArray, vt, ir_node_2_spv_value[type->element_type()],
        spir_builder_->int_immediate_number(spir_builder_->i32_type(),
                                            type->get_constant_shape()[0]));
    ir_node_2_spv_value[type] = vt.id;
    spir_builder_->decorate(spv::OpDecorate, vt, spv::DecorationArrayStride,
                            type->memory_alignment_size(layout_context_));
  }
};

std::unordered_map<const tinyir::Node *, uint32_t> ir_translate_to_spirv(
    const tinyir::Block *blk,
    tinyir::LayoutContext &layout_ctx,
    IRBuilder *spir_builder) {
  Translate2Spirv translator(spir_builder, layout_ctx);
  translator.visit(blk);
  return std::move(translator.ir_node_2_spv_value);
}
const tinyir::Type *translate_ti_type(tinyir::Block &ir_module,
                                      const DataType t,
                                      bool has_buffer_ptr) {
  if (t->is<PrimitiveType>()) {
    return translate_ti_primitive(ir_module, t);
  }
  if (t->is<PointerType>()) {
    if (has_buffer_ptr) {
      return ir_module.emplace_back<IntType>(/*num_bits=*/64,
                                             /*is_signed=*/false);
    } else {
      return ir_module.emplace_back<IntType>(/*num_bits=*/32,
                                             /*is_signed=*/false);
    }
  }
  if (t->is<TensorType>()) {
    return ir_module.emplace_back<ArrayType>(
        translate_ti_primitive(ir_module, t.get_element_type()),
        t->as<TensorType>()->get_num_elements());
  }
  if (auto struct_type = t->cast<lang::StructType>()) {
    std::vector<const tinyir::Type *> element_types;
    auto &elements = struct_type->elements();
    for (auto &element : elements) {
      element_types.push_back(
          translate_ti_type(ir_module, element.type, has_buffer_ptr));
    }
    return ir_module.emplace_back<StructType>(element_types);
  }
  TI_NOT_IMPLEMENTED
}

}  // namespace spirv
}  // namespace taichi::lang
