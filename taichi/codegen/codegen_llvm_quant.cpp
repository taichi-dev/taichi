#ifdef TI_WITH_LLVM
#include "taichi/codegen/codegen_llvm.h"

#include "taichi/ir/statements.h"
#include "taichi/struct/struct_llvm.h"

TLANG_NAMESPACE_BEGIN

namespace {

inline void update_mask(uint64 &mask, uint32 num_bits, uint32 offset) {
  uint64 new_mask =
      (((~(uint64)0) << (64 - num_bits)) >> (64 - offset - num_bits));
  TI_ASSERT((mask & new_mask) == 0);
  mask |= new_mask;
}

}  // namespace

llvm::Value *CodeGenLLVM::atomic_add_quant_int(AtomicOpStmt *stmt,
                                               CustomIntType *cit) {
  auto [byte_ptr, bit_offset] = load_bit_pointer(llvm_val[stmt->dest]);
  auto physical_type = cit->get_physical_type();
  return create_call(
      fmt::format("atomic_add_partial_bits_b{}", data_type_bits(physical_type)),
      {builder->CreateBitCast(byte_ptr, llvm_ptr_type(physical_type)),
       bit_offset, tlctx->get_constant(cit->get_num_bits()),
       builder->CreateIntCast(llvm_val[stmt->val], llvm_type(physical_type),
                              is_signed(stmt->val->ret_type))});
}

llvm::Value *CodeGenLLVM::atomic_add_quant_fixed(AtomicOpStmt *stmt,
                                                 CustomFloatType *cft) {
  auto [byte_ptr, bit_offset] = load_bit_pointer(llvm_val[stmt->dest]);
  auto cit = cft->get_digits_type()->as<CustomIntType>();
  auto val_store = quant_fixed_to_quant_int(cft, cit, llvm_val[stmt->val]);
  auto physical_type = cit->get_physical_type();
  val_store = builder->CreateSExt(val_store, llvm_type(physical_type));

  return create_call(
      fmt::format("atomic_add_partial_bits_b{}", data_type_bits(physical_type)),
      {builder->CreateBitCast(byte_ptr, llvm_ptr_type(physical_type)),
       bit_offset, tlctx->get_constant(cit->get_num_bits()), val_store});
}

llvm::Value *CodeGenLLVM::quant_fixed_to_quant_int(CustomFloatType *cft,
                                                   CustomIntType *cit,
                                                   llvm::Value *real) {
  llvm::Value *s = nullptr;

  // Compute int(real * (1.0 / scale) + 0.5)
  auto s_numeric = 1.0 / cft->get_scale();
  auto compute_type = cft->get_compute_type();
  s = builder->CreateFPCast(tlctx->get_constant(s_numeric),
                            llvm_type(compute_type));
  auto input_real = builder->CreateFPCast(real, llvm_type(compute_type));
  auto scaled = builder->CreateFMul(input_real, s);

  // Add/minus the 0.5 offset for rounding
  scaled = create_call(
      fmt::format("rounding_prepare_f{}", data_type_bits(compute_type)),
      {scaled});

  if (cit->get_is_signed()) {
    return builder->CreateFPToSI(scaled, llvm_type(cit->get_compute_type()));
  } else {
    return builder->CreateFPToUI(scaled, llvm_type(cit->get_compute_type()));
  }
}

void CodeGenLLVM::store_quant_int(llvm::Value *bit_ptr,
                                  CustomIntType *cit,
                                  llvm::Value *value,
                                  bool atomic) {
  auto [byte_ptr, bit_offset] = load_bit_pointer(bit_ptr);
  store_quant_int(byte_ptr, bit_offset, cit, value, atomic);
}

void CodeGenLLVM::store_quant_int(llvm::Value *byte_ptr,
                                  llvm::Value *bit_offset,
                                  CustomIntType *cit,
                                  llvm::Value *value,
                                  bool atomic) {
  // TODO(type): CUDA only supports atomicCAS on 32- and 64-bit integers.
  // Try to support CustomInt/FloatType with 8/16-bit physical
  // types.
  create_call(fmt::format("{}set_partial_bits_b{}", atomic ? "atomic_" : "",
                          data_type_bits(cit->get_physical_type())),
              {builder->CreateBitCast(byte_ptr,
                                      llvm_ptr_type(cit->get_physical_type())),
               bit_offset, tlctx->get_constant(cit->get_num_bits()),
               builder->CreateIntCast(
                   value, llvm_type(cit->get_physical_type()), false)});
}

void CodeGenLLVM::store_masked(llvm::Value *byte_ptr,
                               uint64 mask,
                               Type *physical_type,
                               llvm::Value *value,
                               bool atomic) {
  if (!mask) {
    // do not store anything
    return;
  }
  uint64 full_mask = (~(uint64)0) >> (64 - data_type_bits(physical_type));
  if ((!atomic || prog->config.quant_opt_atomic_demotion) &&
      ((mask & full_mask) == full_mask)) {
    builder->CreateStore(value, byte_ptr);
    return;
  }
  create_call(fmt::format("{}set_mask_b{}", atomic ? "atomic_" : "",
                          data_type_bits(physical_type)),
              {builder->CreateBitCast(byte_ptr, llvm_ptr_type(physical_type)),
               tlctx->get_constant(mask),
               builder->CreateIntCast(value, llvm_type(physical_type), false)});
}

llvm::Value *CodeGenLLVM::get_exponent_offset(llvm::Value *exponent,
                                              CustomFloatType *cft) {
  // Since we have fewer bits in the exponent type than in f32, an
  // offset is necessary to make sure the stored exponent values are
  // representable by the exponent custom int type.
  auto cond = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_NE, exponent,
                                  tlctx->get_constant(0));
  return builder->CreateSelect(
      cond, tlctx->get_constant(cft->get_exponent_conversion_offset()),
      tlctx->get_constant(0));
}

llvm::Value *CodeGenLLVM::quant_int_or_quant_fixed_to_bits(llvm::Value *val,
                                                           Type *input_type,
                                                           Type *output_type) {
  CustomIntType *cit = nullptr;
  if (auto cft = input_type->cast<CustomFloatType>()) {
    TI_ASSERT(cft->get_exponent_type() == nullptr);
    cit = cft->get_digits_type()->as<CustomIntType>();
    val = quant_fixed_to_quant_int(cft, cit, val);
  } else {
    cit = input_type->as<CustomIntType>();
  }
  if (cit->get_num_bits() < val->getType()->getIntegerBitWidth()) {
    val = builder->CreateAnd(
        val, tlctx->get_constant(cit->get_compute_type(),
                                 uint64((1ULL << cit->get_num_bits()) - 1)));
  }
  val = builder->CreateZExt(val, llvm_type(output_type));
  return val;
}

void CodeGenLLVM::visit(BitStructStoreStmt *stmt) {
  auto bit_struct_snode = stmt->get_bit_struct_snode();
  auto bit_struct_physical_type =
      bit_struct_snode->dt->as<BitStructType>()->get_physical_type();

  int bit_struct_num_non_exponent_children = 0;
  for (auto &ch : bit_struct_snode->ch) {
    if (ch->exponent_users.empty()) {
      bit_struct_num_non_exponent_children++;
    }
  }
  bool store_all_components = false;
  if (prog->config.quant_opt_atomic_demotion &&
      stmt->ch_ids.size() == bit_struct_num_non_exponent_children) {
    stmt->is_atomic = false;
    store_all_components = true;
  }

  bool has_shared_exponent = false;
  for (auto ch_id : stmt->ch_ids) {
    if (bit_struct_snode->ch[ch_id]->owns_shared_exponent) {
      has_shared_exponent = true;
    }
  }
  // TODO: what about storing only shared-exponent floating-point SNodes
  //  that don't own the shared exponent?

  if (has_shared_exponent) {
    store_quant_floats_with_shared_exponents(stmt);
  }

  llvm::Value *bit_struct_val = nullptr;
  for (int i = 0; i < stmt->ch_ids.size(); i++) {
    auto ch_id = stmt->ch_ids[i];
    auto val = llvm_val[stmt->values[i]];
    auto &ch = bit_struct_snode->ch[ch_id];
    if (has_shared_exponent && ch->exp_snode != nullptr &&
        ch->exp_snode->exponent_users.size() > 1) {
      // already handled in store_quant_floats_with_shared_exponents
      continue;
    }
    auto dtype = ch->dt;

    if (dtype->is<CustomFloatType>() &&
        dtype->as<CustomFloatType>()->get_exponent_type() != nullptr) {
      // Custom float type with non-shared exponent.
      auto cft = dtype->as<CustomFloatType>();
      llvm::Value *digit_bits = nullptr;
      // Extract exponent and digits from compute type (assumed to be f32 for
      // now).
      TI_ASSERT(cft->get_compute_type()->is_primitive(PrimitiveTypeID::f32));

      // f32 = 1 sign bit + 8 exponent bits + 23 fraction bits

      auto f32_bits =
          builder->CreateBitCast(val, llvm::Type::getInt32Ty(*llvm_context));
      // Rounding to nearest here. Note that if the digits overflows then the
      // carry-on will contribute to the exponent, which is desired.
      if (cft->get_digit_bits() < 23) {
        f32_bits = builder->CreateAdd(
            f32_bits, tlctx->get_constant(1 << (22 - cft->get_digit_bits())));
      }

      auto exponent_bits = builder->CreateAShr(f32_bits, 23);
      exponent_bits =
          builder->CreateAnd(exponent_bits, tlctx->get_constant((1 << 8) - 1));
      auto value_bits = builder->CreateAShr(
          f32_bits, tlctx->get_constant(23 - cft->get_digit_bits()));

      digit_bits = builder->CreateAnd(
          value_bits, tlctx->get_constant((1 << (cft->get_digit_bits())) - 1));

      if (cft->get_is_signed()) {
        // extract the sign bit
        auto sign_bit =
            builder->CreateAnd(f32_bits, tlctx->get_constant(0x80000000u));
        // insert the sign bit to digit bits
        digit_bits = builder->CreateOr(
            digit_bits,
            builder->CreateLShr(sign_bit, 31 - cft->get_digit_bits()));
      }

      auto digits_snode = ch.get();
      auto exponent_snode = digits_snode->exp_snode;

      auto exponent_offset = get_exponent_offset(exponent_bits, cft);
      exponent_bits = builder->CreateSub(exponent_bits, exponent_offset);
      exponent_bits =
          create_call("max_i32", {exponent_bits, tlctx->get_constant(0)});

      // Compute the bit pointer of the exponent bits.
      TI_ASSERT(digits_snode->parent == exponent_snode->parent);

      val = builder->CreateBitCast(exponent_bits,
                                   llvm_type(bit_struct_physical_type));
      val = builder->CreateShl(val, exponent_snode->bit_offset);

      if (bit_struct_val == nullptr) {
        bit_struct_val = val;
      } else {
        bit_struct_val = builder->CreateOr(bit_struct_val, val);
      }
      // Here we implement flush to zero (FTZ): if exponent is zero, we force
      // the digits to be zero.
      // TODO: it seems that this can be more efficiently implemented using a
      // bit_and.
      auto exp_non_zero =
          builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_NE, exponent_bits,
                              tlctx->get_constant(0));
      val = builder->CreateSelect(exp_non_zero, digit_bits,
                                  tlctx->get_constant(0));
      val = builder->CreateBitCast(val, llvm_type(bit_struct_physical_type));
      val = builder->CreateShl(val, digits_snode->bit_offset);
    } else {
      val = quant_int_or_quant_fixed_to_bits(val, dtype,
                                             bit_struct_physical_type);
      val = builder->CreateShl(val, bit_struct_snode->ch[ch_id]->bit_offset);
    }

    if (bit_struct_val == nullptr) {
      bit_struct_val = val;
    } else {
      bit_struct_val = builder->CreateOr(bit_struct_val, val);
    }
  }
  if (store_all_components && !has_shared_exponent) {
    // Store all components here.
    builder->CreateStore(bit_struct_val, llvm_val[stmt->ptr]);
  } else {
    // Create a mask and use a single (atomic)CAS
    uint64 mask = 0;
    for (auto &ch_id : stmt->ch_ids) {
      auto &ch = bit_struct_snode->ch[ch_id];
      if (has_shared_exponent && ch->exp_snode != nullptr &&
          ch->exp_snode->exponent_users.size() > 1) {
        // already handled in store_quant_floats_with_shared_exponents
        continue;
      }
      auto dtype = ch->dt;
      CustomIntType *cit = nullptr;
      if (auto cft = dtype->cast<CustomFloatType>()) {
        if (cft->get_exponent_type() != nullptr) {
          auto exp = cft->get_exponent_type();
          auto exponent_cit = exp->as<CustomIntType>();
          auto exponent_snode = ch->exp_snode;
          update_mask(mask, exponent_cit->get_num_bits(),
                      exponent_snode->bit_offset);
        }
        cit = cft->get_digits_type()->as<CustomIntType>();
      } else {
        cit = dtype->as<CustomIntType>();
      }
      update_mask(mask, cit->get_num_bits(), ch->bit_offset);
    }
    store_masked(llvm_val[stmt->ptr], mask, bit_struct_physical_type,
                 bit_struct_val, stmt->is_atomic);
  }
}

void CodeGenLLVM::store_quant_floats_with_shared_exponents(
    BitStructStoreStmt *stmt) {
  // handle each exponent separately
  auto snode = stmt->get_bit_struct_snode();
  auto bit_struct_physical_type =
      snode->dt->as<BitStructType>()->get_physical_type();
  auto local_bit_struct = builder->CreateLoad(llvm_val[stmt->ptr]);
  // fuse all stores into a masked store
  llvm::Value *masked_val = nullptr;
  uint64 mask = 0;
  for (int i = 0; i < (int)snode->ch.size(); i++) {
    if (snode->ch[i]->exponent_users.empty())
      continue;
    // ch[i] must be an exponent SNode
    auto &exp = snode->ch[i];
    if (exp->exponent_users.size() == 1) {
      // non-shared
      continue;
    }
    // load all floats
    std::vector<llvm::Value *> floats;
    for (auto &user : exp->exponent_users) {
      auto ch_id = snode->child_id(user);
      if (auto input =
              std::find(stmt->ch_ids.begin(), stmt->ch_ids.end(), ch_id);
          input != stmt->ch_ids.end()) {
        floats.push_back(llvm_val[stmt->values[input - stmt->ch_ids.begin()]]);
      } else {
        floats.push_back(extract_quant_float(local_bit_struct, user));
      }
    }
    // convert to i32 for bit operations
    llvm::Value *max_exp_bits = nullptr;
    for (auto f : floats) {
      // TODO: we only support f32 here.
      auto exp_bits = extract_exponent_from_f32(f);
      if (max_exp_bits) {
        max_exp_bits = create_call("max_u32", {max_exp_bits, exp_bits});
      } else {
        max_exp_bits = exp_bits;
      }
    }

    auto first_cft = exp->exponent_users[0]->dt->as<CustomFloatType>();
    auto exponent_offset = get_exponent_offset(max_exp_bits, first_cft);

    auto max_exp_bits_to_store =
        builder->CreateSub(max_exp_bits, exponent_offset);

    max_exp_bits_to_store =
        create_call("max_i32", {max_exp_bits_to_store, tlctx->get_constant(0)});

    // store the exponent
    auto val = builder->CreateZExt(
        max_exp_bits_to_store,
        llvm_type(bit_struct_physical_type->get_compute_type()));
    val = builder->CreateShl(val, exp->bit_offset);
    if (masked_val == nullptr) {
      masked_val = val;
    } else {
      masked_val = builder->CreateOr(masked_val, val);
    }
    update_mask(mask, exp->dt->as<CustomIntType>()->get_num_bits(),
                exp->bit_offset);

    for (int c = 0; c < (int)exp->exponent_users.size(); c++) {
      auto user = exp->exponent_users[c];
      auto ch_id = snode->child_id(user);
      auto digits =
          extract_digits_from_f32_with_shared_exponent(floats[c], max_exp_bits);
      auto digits_snode = snode->ch[ch_id].get();
      auto cft = digits_snode->dt->as<CustomFloatType>();
      auto digits_bit_offset = digits_snode->bit_offset;

      int right_shift_bits = 23 + cft->get_is_signed() - cft->get_digit_bits();
      if (!cft->get_is_signed()) {
        // unsigned
        right_shift_bits += 1;
      }

      // round to nearest
      digits = builder->CreateAdd(
          digits, tlctx->get_constant(1 << (right_shift_bits - 1)));
      // do not allow overflowing
      digits =
          create_call("min_u32", {digits, tlctx->get_constant((1u << 24) - 1)});

      // Compress f32 digits to cft digits.
      // Note that we need to keep the leading 1 bit so 24 instead of 23 in the
      // following code.
      digits = builder->CreateLShr(digits, right_shift_bits);
      if (cft->get_is_signed()) {
        auto float_bits = builder->CreateBitCast(
            floats[c], llvm::Type::getInt32Ty(*llvm_context));
        auto sign_bit = builder->CreateAnd(float_bits, 1 << 31);
        sign_bit = builder->CreateLShr(sign_bit, 31 - cft->get_digit_bits());
        digits = builder->CreateOr(digits, sign_bit);
      }

      // store the digits
      val = builder->CreateZExt(digits, llvm_type(bit_struct_physical_type));
      val = builder->CreateShl(val, digits_bit_offset);
      masked_val = builder->CreateOr(masked_val, val);
      auto num_digit_bits =
          cft->get_digits_type()->as<CustomIntType>()->get_num_bits();
      update_mask(mask, num_digit_bits, digits_bit_offset);
    }
  }
  store_masked(llvm_val[stmt->ptr], mask, bit_struct_physical_type, masked_val,
               stmt->is_atomic);
}

llvm::Value *CodeGenLLVM::extract_exponent_from_f32(llvm::Value *f) {
  TI_ASSERT(f->getType() == llvm::Type::getFloatTy(*llvm_context));
  f = builder->CreateBitCast(f, llvm::Type::getInt32Ty(*llvm_context));
  auto exp_bits = builder->CreateLShr(f, tlctx->get_constant(23));
  return builder->CreateAnd(exp_bits, tlctx->get_constant((1 << 8) - 1));
}

llvm::Value *CodeGenLLVM::extract_digits_from_f32(llvm::Value *f, bool full) {
  TI_ASSERT(f->getType() == llvm::Type::getFloatTy(*llvm_context));
  f = builder->CreateBitCast(f, llvm::Type::getInt32Ty(*llvm_context));
  auto digits = builder->CreateAnd(f, tlctx->get_constant((1 << 23) - 1));
  if (full) {
    digits = builder->CreateOr(digits, tlctx->get_constant(1 << 23));
  }
  return digits;
}

llvm::Value *CodeGenLLVM::extract_digits_from_f32_with_shared_exponent(
    llvm::Value *f,
    llvm::Value *shared_exp) {
  auto exp = extract_exponent_from_f32(f);
  auto exp_offset = builder->CreateSub(shared_exp, exp);
  // TODO: handle negative digits

  // There are two cases that may result in zero digits:
  // - exp is zero. This means f itself is zero. Note that when processors
  // running under FTZ (flush to zero), exp = 0 implies digits = 0.
  // - exp is too small compared to shared_exp, or equivalently exp_offset is
  // too large. This means we need to flush digits to zero.

  // If exp is nonzero, insert an extra "1" bit that was originally implicit.
  auto exp_non_zero = builder->CreateICmpNE(exp, tlctx->get_constant(0));
  exp_non_zero =
      builder->CreateZExt(exp_non_zero, llvm::Type::getInt32Ty(*llvm_context));
  auto implicit_bit = builder->CreateShl(exp_non_zero, tlctx->get_constant(23));

  auto digits = extract_digits_from_f32(f, true);
  digits = builder->CreateOr(digits, implicit_bit);
  exp_offset = create_call("min_u32", {exp_offset, tlctx->get_constant(31)});
  return builder->CreateLShr(digits, exp_offset);
}

llvm::Value *CodeGenLLVM::extract_quant_float(llvm::Value *local_bit_struct,
                                              SNode *digits_snode) {
  auto cft = digits_snode->dt->as<CustomFloatType>();
  auto exponent_type = cft->get_exponent_type()->as<CustomIntType>();
  auto digits_type = cft->get_digits_type()->as<CustomIntType>();
  auto digits = extract_quant_int(local_bit_struct,
                                  tlctx->get_constant(digits_snode->bit_offset),
                                  digits_type);
  auto exponent = extract_quant_int(
      local_bit_struct,
      tlctx->get_constant(digits_snode->exp_snode->bit_offset), exponent_type);
  return reconstruct_quant_float(digits, exponent, cft,
                                 digits_snode->owns_shared_exponent);
}

llvm::Value *CodeGenLLVM::load_quant_int(llvm::Value *ptr, Type *load_type) {
  auto *cit = load_type->as<CustomIntType>();
  auto [byte_ptr, bit_offset] = load_bit_pointer(ptr);

  auto bit_level_container = builder->CreateLoad(builder->CreateBitCast(
      byte_ptr, llvm_ptr_type(cit->get_physical_type())));

  return extract_quant_int(bit_level_container, bit_offset, load_type);
}

llvm::Value *CodeGenLLVM::extract_quant_int(llvm::Value *physical_value,
                                            llvm::Value *bit_offset,
                                            Type *load_type) {
  //  bit shifting
  //    first left shift `physical_type - (offset + num_bits)`
  //    then right shift `physical_type - num_bits`
  auto cit = load_type->as<CustomIntType>();
  auto bit_end =
      builder->CreateAdd(bit_offset, tlctx->get_constant(cit->get_num_bits()));
  auto left = builder->CreateSub(
      tlctx->get_constant(data_type_bits(cit->get_physical_type())), bit_end);
  auto right = builder->CreateSub(
      tlctx->get_constant(data_type_bits(cit->get_physical_type())),
      tlctx->get_constant(cit->get_num_bits()));
  left = builder->CreateIntCast(left, physical_value->getType(), false);
  right = builder->CreateIntCast(right, physical_value->getType(), false);
  auto step1 = builder->CreateShl(physical_value, left);
  llvm::Value *step2 = nullptr;

  if (cit->get_is_signed())
    step2 = builder->CreateAShr(step1, right);
  else
    step2 = builder->CreateLShr(step1, right);

  return builder->CreateIntCast(step2, llvm_type(cit->get_compute_type()),
                                cit->get_is_signed());
}

llvm::Value *CodeGenLLVM::reconstruct_quant_fixed(llvm::Value *digits,
                                                  CustomFloatType *cft) {
  // Compute float(digits) * scale
  llvm::Value *cast = nullptr;
  auto compute_type = cft->get_compute_type()->as<PrimitiveType>();
  if (cft->get_is_signed()) {
    cast = builder->CreateSIToFP(digits, llvm_type(compute_type));
  } else {
    cast = builder->CreateUIToFP(digits, llvm_type(compute_type));
  }
  llvm::Value *s = tlctx->get_constant(cft->get_scale());
  s = builder->CreateFPCast(s, llvm_type(compute_type));
  return builder->CreateFMul(cast, s);
}

llvm::Value *CodeGenLLVM::load_quant_float(llvm::Value *digits_bit_ptr,
                                           llvm::Value *exponent_bit_ptr,
                                           CustomFloatType *cft,
                                           bool shared_exponent) {
  // TODO: we ignore "scale" for CustomFloatType with exponent for now. May need
  // to support this in the future.

  TI_ASSERT(cft->get_scale() == 1);
  auto digits = load_quant_int(digits_bit_ptr, cft->get_digits_type());

  auto exponent_val = load_quant_int(
      exponent_bit_ptr, cft->get_exponent_type()->as<CustomIntType>());
  return reconstruct_quant_float(digits, exponent_val, cft, shared_exponent);
}

llvm::Value *CodeGenLLVM::reconstruct_quant_float(
    llvm::Value *input_digits,
    llvm::Value *input_exponent_val,
    CustomFloatType *cft,
    bool shared_exponent) {
  auto digits = input_digits;
  auto exponent_val = input_exponent_val;
  // Make sure the exponent is within the range of the exponent type
  auto exponent_offset =
      tlctx->get_constant(cft->get_exponent_conversion_offset());

  // Note that zeros need special treatment, when truncated during store.
  auto exponent_type = cft->get_exponent_type()->as<CustomIntType>();
  if (exponent_type->get_num_bits() < 8) {
    auto cond = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_NE,
                                    exponent_val, tlctx->get_constant(0));
    exponent_offset =
        builder->CreateSelect(cond, exponent_offset, tlctx->get_constant(0));
  }

  if (cft->get_compute_type()->is_primitive(PrimitiveTypeID::f32)) {
    // Construct an f32 out of exponent_val and digits
    // Assuming digits and exponent_val are i32
    // f32 = 1 sign bit + 8 exponent bits + 23 fraction bits

    digits = builder->CreateAnd(
        digits,
        (1u << cft->get_digits_type()->as<CustomIntType>()->get_num_bits()) -
            1);

    llvm::Value *sign_bit = nullptr;

    if (shared_exponent) {
      if (cft->get_is_signed()) {
        sign_bit = builder->CreateAnd(
            digits, tlctx->get_constant(1u << cft->get_digit_bits()));
        digits = builder->CreateXor(digits, sign_bit);
        sign_bit = builder->CreateShl(sign_bit, 31 - cft->get_digit_bits());
        digits = builder->CreateShl(digits, 1);
      }
      // There is a leading 1 that marks the beginning of the digits.
      // When not using shared exponents, the 1 bit is not needed (since digits
      // always starts with 1).
      // declare i32  @llvm.ctlz.i32 (i32  <src>, i1 <is_zero_undef>)
      auto num_leading_zeros = builder->CreateIntrinsic(
          llvm::Intrinsic::ctlz, {llvm::Type::getInt32Ty(*llvm_context)},
          {digits, tlctx->get_constant(false)});
      auto extra_shift = builder->CreateSub(
          tlctx->get_constant(31 - cft->get_digit_bits()), num_leading_zeros);
      exponent_offset = builder->CreateAdd(exponent_offset, extra_shift);

      if (!cft->get_is_signed())
        exponent_offset =
            builder->CreateAdd(exponent_offset, tlctx->get_constant(1));

      auto digits_shift = builder->CreateSub(
          tlctx->get_constant(23 - cft->get_digit_bits()), extra_shift);
      digits = builder->CreateShl(digits, digits_shift);
    } else {
      digits = builder->CreateShl(
          digits, tlctx->get_constant(23 - cft->get_digit_bits()));
    }
    auto fraction_bits = builder->CreateAnd(digits, (1u << 23) - 1);

    exponent_val = builder->CreateAdd(exponent_val, exponent_offset);

    auto exponent_bits =
        builder->CreateShl(exponent_val, tlctx->get_constant(23));

    auto f32_bits = builder->CreateOr(exponent_bits, fraction_bits);

    if (shared_exponent) {
      // Handle zero exponent
      auto zero_exponent =
          builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_EQ,
                              input_exponent_val, tlctx->get_constant(0));
      auto zero_digits =
          builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_EQ, input_digits,
                              tlctx->get_constant(0));
      auto zero_output = builder->CreateOr(zero_exponent, zero_digits);
      f32_bits =
          builder->CreateSelect(zero_output, tlctx->get_constant(0), f32_bits);
    }

    if (cft->get_is_signed()) {
      if (!sign_bit) {
        sign_bit = builder->CreateAnd(digits, tlctx->get_constant(1u << 23));
        sign_bit = builder->CreateShl(sign_bit, tlctx->get_constant(31 - 23));
      }
      f32_bits = builder->CreateOr(f32_bits, sign_bit);
    }

    return builder->CreateBitCast(f32_bits,
                                  llvm::Type::getFloatTy(*llvm_context));
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

llvm::Value *CodeGenLLVM::load_quant_fixed_or_quant_float(Stmt *ptr_stmt) {
  auto ptr = ptr_stmt->as<GetChStmt>();
  auto cft = ptr->ret_type->as<PointerType>()
                 ->get_pointee_type()
                 ->as<CustomFloatType>();
  if (cft->get_exponent_type()) {
    TI_ASSERT(ptr->width() == 1);
    auto digits_bit_ptr = llvm_val[ptr];
    auto digits_snode = ptr->output_snode;
    auto exponent_snode = digits_snode->exp_snode;
    // Compute the bit pointer of the exponent bits.
    TI_ASSERT(digits_snode->parent == exponent_snode->parent);
    auto exponent_bit_ptr = offset_bit_ptr(
        digits_bit_ptr, exponent_snode->bit_offset - digits_snode->bit_offset);
    return load_quant_float(digits_bit_ptr, exponent_bit_ptr, cft,
                            digits_snode->owns_shared_exponent);
  } else {
    auto digits = load_quant_int(llvm_val[ptr], cft->get_digits_type());
    return reconstruct_quant_fixed(digits, cft);
  }
}

TLANG_NAMESPACE_END

#endif  // #ifdef TI_WITH_LLVM
