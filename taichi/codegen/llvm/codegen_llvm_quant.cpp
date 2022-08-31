#ifdef TI_WITH_LLVM
#include "taichi/codegen/llvm/codegen_llvm.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

namespace {

inline void update_mask(uint64 &mask, uint32 num_bits, uint32 offset) {
  uint64 new_mask =
      (((~(uint64)0) << (64 - num_bits)) >> (64 - offset - num_bits));
  TI_ASSERT((mask & new_mask) == 0);
  mask |= new_mask;
}

}  // namespace

llvm::Value *TaskCodeGenLLVM::atomic_add_quant_int(llvm::Value *ptr,
                                                   llvm::Type *physical_type,
                                                   QuantIntType *qit,
                                                   llvm::Value *value,
                                                   bool value_is_signed) {
  auto [byte_ptr, bit_offset] = load_bit_ptr(ptr);
  return create_call(
      fmt::format("atomic_add_partial_bits_b{}",
                  physical_type->getIntegerBitWidth()),
      {byte_ptr, bit_offset, tlctx->get_constant(qit->get_num_bits()),
       builder->CreateIntCast(value, physical_type, value_is_signed)});
}

llvm::Value *TaskCodeGenLLVM::atomic_add_quant_fixed(llvm::Value *ptr,
                                                     llvm::Type *physical_type,
                                                     QuantFixedType *qfxt,
                                                     llvm::Value *value) {
  auto [byte_ptr, bit_offset] = load_bit_ptr(ptr);
  auto qit = qfxt->get_digits_type()->as<QuantIntType>();
  auto val_store = to_quant_fixed(value, qfxt);
  val_store = builder->CreateSExt(val_store, physical_type);
  return create_call(fmt::format("atomic_add_partial_bits_b{}",
                                 physical_type->getIntegerBitWidth()),
                     {byte_ptr, bit_offset,
                      tlctx->get_constant(qit->get_num_bits()), val_store});
}

llvm::Value *TaskCodeGenLLVM::to_quant_fixed(llvm::Value *real,
                                             QuantFixedType *qfxt) {
  // Compute int(real * (1.0 / scale) + 0.5)
  auto compute_type = qfxt->get_compute_type();
  auto s = builder->CreateFPCast(tlctx->get_constant(1.0 / qfxt->get_scale()),
                                 tlctx->get_data_type(compute_type));
  auto input_real =
      builder->CreateFPCast(real, tlctx->get_data_type(compute_type));
  auto scaled = builder->CreateFMul(input_real, s);

  // Add/minus the 0.5 offset for rounding
  scaled = create_call(
      fmt::format("rounding_prepare_f{}", data_type_bits(compute_type)),
      {scaled});

  auto qit = qfxt->get_digits_type()->as<QuantIntType>();
  if (qit->get_is_signed()) {
    return builder->CreateFPToSI(scaled,
                                 tlctx->get_data_type(qit->get_compute_type()));
  } else {
    return builder->CreateFPToUI(scaled,
                                 tlctx->get_data_type(qit->get_compute_type()));
  }
}

void TaskCodeGenLLVM::store_quant_int(llvm::Value *ptr,
                                      llvm::Type *physical_type,
                                      QuantIntType *qit,
                                      llvm::Value *value,
                                      bool atomic) {
  auto [byte_ptr, bit_offset] = load_bit_ptr(ptr);
  // TODO(type): CUDA only supports atomicCAS on 32- and 64-bit integers.
  // Try to support 8/16-bit physical types.
  create_call(fmt::format("{}set_partial_bits_b{}", atomic ? "atomic_" : "",
                          physical_type->getIntegerBitWidth()),
              {byte_ptr, bit_offset, tlctx->get_constant(qit->get_num_bits()),
               builder->CreateIntCast(value, physical_type, false)});
}

void TaskCodeGenLLVM::store_quant_fixed(llvm::Value *ptr,
                                        llvm::Type *physical_type,
                                        QuantFixedType *qfxt,
                                        llvm::Value *value,
                                        bool atomic) {
  store_quant_int(ptr, physical_type,
                  qfxt->get_digits_type()->as<QuantIntType>(),
                  to_quant_fixed(value, qfxt), atomic);
}

void TaskCodeGenLLVM::store_masked(llvm::Value *ptr,
                                   llvm::Type *ty,
                                   uint64 mask,
                                   llvm::Value *value,
                                   bool atomic) {
  if (!mask) {
    // do not store anything
    return;
  }
  uint64 full_mask = (~(uint64)0) >> (64 - ty->getIntegerBitWidth());
  if ((!atomic || prog->config.quant_opt_atomic_demotion) &&
      ((mask & full_mask) == full_mask)) {
    builder->CreateStore(value, ptr);
    return;
  }
  create_call(fmt::format("{}set_mask_b{}", atomic ? "atomic_" : "",
                          ty->getIntegerBitWidth()),
              {ptr, tlctx->get_constant(mask),
               builder->CreateIntCast(value, ty, false)});
}

llvm::Value *TaskCodeGenLLVM::get_exponent_offset(llvm::Value *exponent,
                                                  QuantFloatType *qflt) {
  // Since we have fewer bits in the exponent type than in f32, an
  // offset is necessary to make sure the stored exponent values are
  // representable by the exponent quant int type.
  auto cond = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_NE, exponent,
                                  tlctx->get_constant(0));
  return builder->CreateSelect(
      cond, tlctx->get_constant(qflt->get_exponent_conversion_offset()),
      tlctx->get_constant(0));
}

llvm::Value *TaskCodeGenLLVM::quant_int_or_quant_fixed_to_bits(
    llvm::Value *val,
    Type *input_type,
    llvm::Type *output_type) {
  QuantIntType *qit = nullptr;
  if (auto qfxt = input_type->cast<QuantFixedType>()) {
    qit = qfxt->get_digits_type()->as<QuantIntType>();
    val = to_quant_fixed(val, qfxt);
  } else {
    qit = input_type->as<QuantIntType>();
  }
  if (qit->get_num_bits() < val->getType()->getIntegerBitWidth()) {
    val = builder->CreateAnd(
        val, tlctx->get_constant(qit->get_compute_type(),
                                 uint64((1ULL << qit->get_num_bits()) - 1)));
  }
  val = builder->CreateZExt(val, output_type);
  return val;
}

void TaskCodeGenLLVM::visit(BitStructStoreStmt *stmt) {
  auto bit_struct = stmt->get_bit_struct();
  auto physical_type = tlctx->get_data_type(bit_struct->get_physical_type());

  int num_non_exponent_children = 0;
  for (int i = 0; i < bit_struct->get_num_members(); i++) {
    if (bit_struct->get_member_exponent_users(i).empty()) {
      num_non_exponent_children++;
    }
  }
  bool store_all_components = false;
  if (prog->config.quant_opt_atomic_demotion &&
      stmt->ch_ids.size() == num_non_exponent_children) {
    stmt->is_atomic = false;
    store_all_components = true;
  }

  bool has_shared_exponent = false;
  for (auto ch_id : stmt->ch_ids) {
    if (bit_struct->get_member_owns_shared_exponent(ch_id)) {
      has_shared_exponent = true;
    }
  }
  if (has_shared_exponent) {
    store_quant_floats_with_shared_exponents(stmt);
  }

  llvm::Value *bit_struct_val = nullptr;
  for (int i = 0; i < stmt->ch_ids.size(); i++) {
    auto ch_id = stmt->ch_ids[i];
    auto exp = bit_struct->get_member_exponent(ch_id);
    if (exp != -1 && bit_struct->get_member_exponent_users(exp).size() > 1) {
      // already handled in store_quant_floats_with_shared_exponents
      continue;
    }
    auto dtype = bit_struct->get_member_type(ch_id);
    auto val = llvm_val[stmt->values[i]];
    if (auto qflt = dtype->cast<QuantFloatType>()) {
      // Quant float type with non-shared exponent.
      llvm::Value *digit_bits = nullptr;
      // Extract exponent and digits from compute type (assumed to be f32 for
      // now).
      TI_ASSERT(qflt->get_compute_type()->is_primitive(PrimitiveTypeID::f32));

      // f32 = 1 sign bit + 8 exponent bits + 23 fraction bits

      auto f32_bits =
          builder->CreateBitCast(val, llvm::Type::getInt32Ty(*llvm_context));
      // Rounding to nearest here. Note that if the digits overflows then the
      // carry-on will contribute to the exponent, which is desired.
      if (qflt->get_digit_bits() < 23) {
        f32_bits = builder->CreateAdd(
            f32_bits, tlctx->get_constant(1 << (22 - qflt->get_digit_bits())));
      }

      auto exponent_bits = builder->CreateAShr(f32_bits, 23);
      exponent_bits =
          builder->CreateAnd(exponent_bits, tlctx->get_constant((1 << 8) - 1));
      auto value_bits = builder->CreateAShr(
          f32_bits, tlctx->get_constant(23 - qflt->get_digit_bits()));

      digit_bits = builder->CreateAnd(
          value_bits, tlctx->get_constant((1 << (qflt->get_digit_bits())) - 1));

      if (qflt->get_is_signed()) {
        // extract the sign bit
        auto sign_bit =
            builder->CreateAnd(f32_bits, tlctx->get_constant(0x80000000u));
        // insert the sign bit to digit bits
        digit_bits = builder->CreateOr(
            digit_bits,
            builder->CreateLShr(sign_bit, 31 - qflt->get_digit_bits()));
      }

      auto exponent_offset = get_exponent_offset(exponent_bits, qflt);
      exponent_bits = builder->CreateSub(exponent_bits, exponent_offset);
      exponent_bits =
          create_call("max_i32", {exponent_bits, tlctx->get_constant(0)});

      // Compute the bit pointer of the exponent bits.
      val = builder->CreateIntCast(exponent_bits, physical_type, false);
      val = builder->CreateShl(val, bit_struct->get_member_bit_offset(exp));

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
      val = builder->CreateIntCast(val, physical_type, false);
      val = builder->CreateShl(val, bit_struct->get_member_bit_offset(ch_id));
    } else {
      val = quant_int_or_quant_fixed_to_bits(val, dtype, physical_type);
      val = builder->CreateShl(val, bit_struct->get_member_bit_offset(ch_id));
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
    for (int i = 0; i < stmt->ch_ids.size(); i++) {
      auto ch_id = stmt->ch_ids[i];
      auto exp = bit_struct->get_member_exponent(ch_id);
      if (exp != -1 && bit_struct->get_member_exponent_users(exp).size() > 1) {
        // already handled in store_quant_floats_with_shared_exponents
        continue;
      }
      auto dtype = bit_struct->get_member_type(ch_id);
      QuantIntType *qit = nullptr;
      if (auto qflt = dtype->cast<QuantFloatType>()) {
        auto exponent_qit = qflt->get_exponent_type()->as<QuantIntType>();
        update_mask(mask, exponent_qit->get_num_bits(),
                    bit_struct->get_member_bit_offset(exp));
        qit = qflt->get_digits_type()->as<QuantIntType>();
      } else if (auto qfxt = dtype->cast<QuantFixedType>()) {
        qit = qfxt->get_digits_type()->as<QuantIntType>();
      } else {
        qit = dtype->as<QuantIntType>();
      }
      update_mask(mask, qit->get_num_bits(),
                  bit_struct->get_member_bit_offset(ch_id));
    }
    store_masked(llvm_val[stmt->ptr], physical_type, mask, bit_struct_val,
                 stmt->is_atomic);
  }
}

void TaskCodeGenLLVM::store_quant_floats_with_shared_exponents(
    BitStructStoreStmt *stmt) {
  // handle each exponent separately
  auto bit_struct = stmt->get_bit_struct();
  auto physical_type = tlctx->get_data_type(bit_struct->get_physical_type());
  auto physical_value = builder->CreateLoad(physical_type, llvm_val[stmt->ptr]);
  // fuse all stores into a masked store
  llvm::Value *masked_val = nullptr;
  uint64 mask = 0;
  for (int i = 0; i < bit_struct->get_num_members(); i++) {
    auto &exponent_users = bit_struct->get_member_exponent_users(i);
    // make sure i-th member is a shared exponent
    if (exponent_users.size() < 2)
      continue;
    // load all floats with the shared exponent
    std::vector<llvm::Value *> floats;
    for (auto user : exponent_users) {
      if (auto input =
              std::find(stmt->ch_ids.begin(), stmt->ch_ids.end(), user);
          input != stmt->ch_ids.end()) {
        floats.push_back(llvm_val[stmt->values[input - stmt->ch_ids.begin()]]);
      } else {
        floats.push_back(extract_quant_float(physical_value, bit_struct, user));
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

    auto first_qflt =
        bit_struct->get_member_type(exponent_users[0])->as<QuantFloatType>();
    auto exponent_offset = get_exponent_offset(max_exp_bits, first_qflt);

    auto max_exp_bits_to_store =
        builder->CreateSub(max_exp_bits, exponent_offset);

    max_exp_bits_to_store =
        create_call("max_i32", {max_exp_bits_to_store, tlctx->get_constant(0)});

    // store the exponent
    auto bit_offset = bit_struct->get_member_bit_offset(i);
    auto val = builder->CreateZExt(max_exp_bits_to_store, physical_type);
    val = builder->CreateShl(val, bit_offset);
    if (masked_val == nullptr) {
      masked_val = val;
    } else {
      masked_val = builder->CreateOr(masked_val, val);
    }
    update_mask(
        mask,
        bit_struct->get_member_type(i)->as<QuantIntType>()->get_num_bits(),
        bit_offset);

    for (int c = 0; c < (int)exponent_users.size(); c++) {
      auto user = exponent_users[c];
      auto digits =
          extract_digits_from_f32_with_shared_exponent(floats[c], max_exp_bits);
      auto qflt = bit_struct->get_member_type(user)->as<QuantFloatType>();
      auto digits_bit_offset = bit_struct->get_member_bit_offset(user);
      auto right_shift_bits = 24 - qflt->get_digit_bits();

      // round to nearest
      digits = builder->CreateAdd(
          digits, tlctx->get_constant(1 << (right_shift_bits - 1)));
      // do not allow overflowing
      digits =
          create_call("min_u32", {digits, tlctx->get_constant((1u << 24) - 1)});

      // Compress f32 digits to qflt digits.
      // Note that we need to keep the leading 1 bit so 24 instead of 23 in the
      // following code.
      digits = builder->CreateLShr(digits, right_shift_bits);
      if (qflt->get_is_signed()) {
        auto float_bits = builder->CreateBitCast(
            floats[c], llvm::Type::getInt32Ty(*llvm_context));
        auto sign_bit = builder->CreateAnd(float_bits, 1 << 31);
        sign_bit = builder->CreateLShr(sign_bit, 31 - qflt->get_digit_bits());
        digits = builder->CreateOr(digits, sign_bit);
      }

      // store the digits
      val = builder->CreateZExt(digits, physical_type);
      val = builder->CreateShl(val, digits_bit_offset);
      masked_val = builder->CreateOr(masked_val, val);
      auto num_digit_bits =
          qflt->get_digits_type()->as<QuantIntType>()->get_num_bits();
      update_mask(mask, num_digit_bits, digits_bit_offset);
    }
  }
  store_masked(llvm_val[stmt->ptr], physical_type, mask, masked_val,
               stmt->is_atomic);
}

llvm::Value *TaskCodeGenLLVM::extract_exponent_from_f32(llvm::Value *f) {
  TI_ASSERT(f->getType() == llvm::Type::getFloatTy(*llvm_context));
  f = builder->CreateBitCast(f, llvm::Type::getInt32Ty(*llvm_context));
  auto exp_bits = builder->CreateLShr(f, tlctx->get_constant(23));
  return builder->CreateAnd(exp_bits, tlctx->get_constant((1 << 8) - 1));
}

llvm::Value *TaskCodeGenLLVM::extract_digits_from_f32(llvm::Value *f,
                                                      bool full) {
  TI_ASSERT(f->getType() == llvm::Type::getFloatTy(*llvm_context));
  f = builder->CreateBitCast(f, llvm::Type::getInt32Ty(*llvm_context));
  auto digits = builder->CreateAnd(f, tlctx->get_constant((1 << 23) - 1));
  if (full) {
    digits = builder->CreateOr(digits, tlctx->get_constant(1 << 23));
  }
  return digits;
}

llvm::Value *TaskCodeGenLLVM::extract_digits_from_f32_with_shared_exponent(
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

llvm::Value *TaskCodeGenLLVM::extract_quant_float(llvm::Value *physical_value,
                                                  BitStructType *bit_struct,
                                                  int digits_id) {
  auto qflt = bit_struct->get_member_type(digits_id)->as<QuantFloatType>();
  auto exponent_id = bit_struct->get_member_exponent(digits_id);
  auto exponent_bit_offset = bit_struct->get_member_bit_offset(exponent_id);
  auto digits_bit_offset = bit_struct->get_member_bit_offset(digits_id);
  auto shared_exponent = bit_struct->get_member_owns_shared_exponent(digits_id);
  auto digits =
      extract_quant_int(physical_value, tlctx->get_constant(digits_bit_offset),
                        qflt->get_digits_type()->as<QuantIntType>());
  auto exponent = extract_quant_int(
      physical_value, tlctx->get_constant(exponent_bit_offset),
      qflt->get_exponent_type()->as<QuantIntType>());
  return reconstruct_quant_float(digits, exponent, qflt, shared_exponent);
}

llvm::Value *TaskCodeGenLLVM::extract_quant_int(llvm::Value *physical_value,
                                                llvm::Value *bit_offset,
                                                QuantIntType *qit) {
  auto physical_type = physical_value->getType();
  //  bit shifting
  //    first left shift `physical_type - (offset + num_bits)`
  //    then right shift `physical_type - num_bits`
  auto bit_end =
      builder->CreateAdd(bit_offset, tlctx->get_constant(qit->get_num_bits()));
  auto left = builder->CreateSub(
      tlctx->get_constant(physical_type->getIntegerBitWidth()), bit_end);
  auto right = builder->CreateSub(
      tlctx->get_constant(physical_type->getIntegerBitWidth()),
      tlctx->get_constant(qit->get_num_bits()));
  left = builder->CreateIntCast(left, physical_type, false);
  right = builder->CreateIntCast(right, physical_type, false);
  auto step1 = builder->CreateShl(physical_value, left);
  llvm::Value *step2 = nullptr;

  if (qit->get_is_signed())
    step2 = builder->CreateAShr(step1, right);
  else
    step2 = builder->CreateLShr(step1, right);

  return builder->CreateIntCast(step2,
                                tlctx->get_data_type(qit->get_compute_type()),
                                qit->get_is_signed());
}

llvm::Value *TaskCodeGenLLVM::reconstruct_quant_fixed(llvm::Value *digits,
                                                      QuantFixedType *qfxt) {
  // Compute float(digits) * scale
  llvm::Value *cast = nullptr;
  auto compute_type = qfxt->get_compute_type()->as<PrimitiveType>();
  if (qfxt->get_is_signed()) {
    cast = builder->CreateSIToFP(digits, tlctx->get_data_type(compute_type));
  } else {
    cast = builder->CreateUIToFP(digits, tlctx->get_data_type(compute_type));
  }
  llvm::Value *s = tlctx->get_constant(qfxt->get_scale());
  s = builder->CreateFPCast(s, tlctx->get_data_type(compute_type));
  return builder->CreateFMul(cast, s);
}

llvm::Value *TaskCodeGenLLVM::reconstruct_quant_float(
    llvm::Value *input_digits,
    llvm::Value *input_exponent_val,
    QuantFloatType *qflt,
    bool shared_exponent) {
  auto digits = input_digits;
  auto exponent_val = input_exponent_val;
  // Make sure the exponent is within the range of the exponent type
  auto exponent_offset =
      tlctx->get_constant(qflt->get_exponent_conversion_offset());

  // Note that zeros need special treatment, when truncated during store.
  auto exponent_type = qflt->get_exponent_type()->as<QuantIntType>();
  if (exponent_type->get_num_bits() < 8) {
    auto cond = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_NE,
                                    exponent_val, tlctx->get_constant(0));
    exponent_offset =
        builder->CreateSelect(cond, exponent_offset, tlctx->get_constant(0));
  }

  if (qflt->get_compute_type()->is_primitive(PrimitiveTypeID::f32)) {
    // Construct an f32 out of exponent_val and digits
    // Assuming digits and exponent_val are i32
    // f32 = 1 sign bit + 8 exponent bits + 23 fraction bits

    digits = builder->CreateAnd(
        digits,
        (1u << qflt->get_digits_type()->as<QuantIntType>()->get_num_bits()) -
            1);

    llvm::Value *sign_bit = nullptr;

    if (shared_exponent) {
      if (qflt->get_is_signed()) {
        sign_bit = builder->CreateAnd(
            digits, tlctx->get_constant(1u << qflt->get_digit_bits()));
        digits = builder->CreateXor(digits, sign_bit);
        sign_bit = builder->CreateShl(sign_bit, 31 - qflt->get_digit_bits());
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
          tlctx->get_constant(31 - qflt->get_digit_bits()), num_leading_zeros);
      exponent_offset = builder->CreateAdd(exponent_offset, extra_shift);

      if (!qflt->get_is_signed())
        exponent_offset =
            builder->CreateAdd(exponent_offset, tlctx->get_constant(1));

      auto digits_shift = builder->CreateSub(
          tlctx->get_constant(23 - qflt->get_digit_bits()), extra_shift);
      digits = builder->CreateShl(digits, digits_shift);
    } else {
      digits = builder->CreateShl(
          digits, tlctx->get_constant(23 - qflt->get_digit_bits()));
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

    if (qflt->get_is_signed()) {
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

TLANG_NAMESPACE_END

#endif  // #ifdef TI_WITH_LLVM
