#include "codegen_cpu.h"
#include "program.h"
#include <taichi/common/bit.h>

TLANG_NAMESPACE_BEGIN

void CPUCodeGen::visit_intrinsics(Expr &expr) {
  // TC_P(expr->id);
  // TC_P(expr->node_type_name());
  // TC_P(num_groups);
  auto vv_width = expr->lanes;
  TC_ASSERT(vv_width % simd_width == 0);
  int split = vv_width / simd_width;
  auto vv_type = [&](DataType dt) {
    return fmt::format("vvec<{}, {}, {}>", data_type_name(dt), simd_width,
                       split);
  };
  if (expr->type == NodeType::addr) {
    return;
  }
  if (expr->var_name == "") {
    // expr->var_name = create_variable();
    expr->var_name = fmt::format("var{}", expr->id);
    /*
    TC_INFO("{} {} {} -> {}", expr->id, expr->node_type_name(),
            expr->data_type_name(), expr->var_name);
            */
  } else
    return;  // visited

  for (auto &m : expr->members) {
    if (!m->name().empty()) {
      emit_code("// @ {}", m->name());
    }
  }

  if (expr->type == NodeType::binary) {
    auto op = binary_type_name(expr->binary_type);
    emit_code("auto {} = {}({}, {});", expr->var_name, op,
              expr->ch[0]->var_name, expr->ch[1]->var_name);
    // emit_code("{}.print();", expr->var_name);
  } else if (expr->type == NodeType::land) {
    TC_ASSERT(expr[1]->type == NodeType::imm)
    TC_WARN("member imm");
    emit_code("auto {} = land({}, {});", expr->var_name, expr[0]->var_name,
              expr[1]->value<int>());
  } else if (expr->type == NodeType::shr) {
    TC_WARN("member imm");
    TC_ASSERT(expr[1]->type == NodeType::imm)
    emit_code("auto {} = shr({}, {});", expr->var_name, expr[0]->var_name,
              expr[1]->value<int>());
  } else if (expr->type == NodeType::shl) {
    TC_WARN("member imm");
    TC_ASSERT(expr[1]->type == NodeType::imm)
    emit_code("auto {} = shl({}, {});", expr->var_name, expr[0]->var_name,
              expr[1]->value<int>());
  } else if (expr->type == NodeType::cmp) {
    auto t = expr->value<CmpType>();
    TC_P((int)t);
    if (t == CmpType::ne) {
      emit_code("auto {} = cmp_ne({}, {});", expr->var_name, expr[0]->var_name,
                expr[1]->var_name);
    } else if (t == CmpType::lt) {
      emit_code("auto {} = cmp_lt({}, {});", expr->var_name, expr[0]->var_name,
                expr[1]->var_name);
    } else {
      TC_NOT_IMPLEMENTED
    }
  } else if (expr->type == NodeType::select) {
    emit_code("auto {} = select({}, {}, {});", expr->var_name,
              expr[0]->var_name, expr[1]->var_name, expr[2]->var_name);
  } else if (expr->type == NodeType::floor) {
    emit_code("auto {} = floor({});", expr->var_name, expr[0]->var_name);
  } else if (expr->type == NodeType::cast) {
    if (expr->data_type == DataType::i32) {
      emit_code("auto {} = {}.cast<int32>();", expr->var_name,
                expr[0]->var_name);
    } else if (expr->data_type == DataType::f32) {
      emit_code("auto {} = {}.cast<float32>();", expr->var_name,
                expr[0]->var_name);
    } else {
      TC_NOT_IMPLEMENTED
    }
  } else if (expr->type == NodeType::vload) {
    emit_code(
        "auto {} = {}::load(access_{}(context.buffers[0], {}.element(0)));",
        expr->var_name, vv_type(expr->data_type),
        expr[0]->new_addresses(0)->node_type_name, expr[1]->var_name);
  } else if (expr->type == NodeType::vstore) {
    emit_code("{}.store(access_{}(context.buffers[0], {}.element(0)));",
              expr[2]->var_name, expr[0]->new_addresses(0)->node_type_name,
              expr[1]->var_name);
  } else if (expr->type == NodeType::load) {
    emit_code("auto {} = {}::load({});", expr->var_name,
              vv_type(expr->data_type), expr[0]->var_name);
#if (0)
    bool regular = false;
    if (expr[0]->type == NodeType::pointer &&
        expr[0][1]->type == NodeType::index) {
      regular = true;
    } else {
    }
    if (regular) {
      /*
      emit_code("auto {} = load<{}, {}>({}_base, {}_offsets);", expr->var_name,
                vv_width, data_type_name(expr->data_type), expr[0]->var_name,
                expr[0]->var_name);
                */
      // TC_P(expr->members.size());

      std::vector<int> offsets;
      for (int i = 0; i + 1 < (int)expr->members.size(); i++) {
        TC_ASSERT(
            expr->members[i]->addr().same_type(expr->members[i + 1]->addr()));
      }
      for (int i = 0; i < (int)expr->members.size(); i++) {
        offsets.push_back(expr->members[i]->addr().offset());
        TC_P(expr->members[i]->addr());
      }
      auto addr = expr->addr();
      // TC_P(i_stride);
      // TC_P(addr.coeff_aosoa_group_size);
      TC_ASSERT(addr.coeff_aosoa_group_size == 0 ||
                num_groups == addr.coeff_aosoa_group_size);
      // TC_ASSERT(expr->members[0]->addr.coeff_i);
      std::string load_instr = "_mm256_load_ps";
      bool needs_shuffle = false;
      if (addr.coeff_const % simd_width != 0) {
        addr.coeff_const -= addr.coeff_const % simd_width;
        needs_shuffle = true;
      }
      for (int i = 0; i < offsets.size(); i++) {
        TC_P(offsets[i]);
      }
      emit_code("auto {} = vvec<{}, {}, {}>::load({});", expr->var_name,
                expr->data_type_name(), simd_width, split,
                get_vectorized_address(addr, 0,
                                       offsets[0] / simd_width * simd_width));
      auto emit_shuffle = [&](std::string imm) {
        for (int i = 0; i < split; i++) {
          emit_code(
              "{}.d[{}] = _mm256_shuffle_ps({}.d[{}], {}.d[{}], "
              "{});",
              expr->var_name, i, expr->var_name, i, expr->var_name, i, imm);
        }
        needs_shuffle = false;
      };
      if (group_size == 1) {
        // emit_code("{}.d[0] = {}.d[0];", expr->var_name, expr->var_name);
      } else {
        TC_ASSERT(group_size <= 8);
        // detect patterns
        int offset_const = offsets[0] % simd_width;
        int offset_inc = offsets[1] - offsets[0];
        if (group_size == 2) {
          if (offset_const == 0 && offset_inc == 1) {
            // emit_code("{}.d[0] = {}.d[0];", expr->var_name, expr->var_name);
          } else if (offset_inc == 0) {
            if (offset_const == 0) {
              emit_shuffle("0xA0");
            } else if (offset_const == 1) {
              emit_shuffle("0xF5");
            } else {
              TC_NOT_IMPLEMENTED;
            }
          } else {
            TC_P(offset_const);
            TC_P(offset_inc);
            TC_NOT_IMPLEMENTED;
          }
        } else if (group_size == 4) {
          if (offset_const == 0 && offset_inc == 1) {
            // emit_code("{}.d[0] = {}.d[0];", expr->var_name, expr->var_name);
          } else if (offset_inc == 0) {
            if (offset_const == 0) {
              emit_shuffle("0x00");
            } else if (offset_const == 1) {
              emit_shuffle("0x55");
            } else if (offset_const == 2) {
              emit_shuffle("0xAA");
            } else if (offset_const == 3) {
              emit_shuffle("0xFF");
            } else {
              TC_NOT_IMPLEMENTED;
            }
          } else {
            TC_P(offset_const);
            TC_P(offset_inc);
            TC_NOT_IMPLEMENTED;
          }
        } else if (group_size == 8) {
          if (offset_inc == 1) {
            TC_ASSERT(offset_const == 0);
            // emit_code("{}.d[0] = {}.d[0];", expr->var_name, expr->var_name);
          } else {
            TC_ASSERT(offset_inc == 0);
            needs_shuffle = false;
            for (int i = 0; i < split; i++)
              emit_code("{}.d[{}] = _mm256_broadcast_ss({});", expr->var_name,
                        i, get_vectorized_address(expr->addr(), 0, 0));
          }
        } else {
          TC_NOT_IMPLEMENTED
        }
        TC_ASSERT(needs_shuffle == false);
      }
    } else {
      // irregular
      emit_code("auto {} = {}::load({}_base, {}_offsets);", expr->var_name,
                vv_type(expr->data_type), expr[0]->var_name, expr[0]->var_name);
    }
#endif
  } else if (expr->type == NodeType::store) {
    emit_code("{}.store({});", expr->ch[1]->var_name, expr[0]->var_name);
  } else if (expr->type == NodeType::combine) {
    // do nothing
  } else if (expr->type == NodeType::imm) {
    TC_WARN("Using member imm");
    if (expr->data_type == DataType::i32) {
      emit_code("auto {} = vvec<int32, {}, {}>({}); /*i32*/ ", expr->var_name,
                simd_width, split, expr->value<int32>());
    } else {
      emit_code("auto {} = vvec<float32, {}, {}>({}); /*f32*/ ", expr->var_name,
                simd_width, split, expr->value<float32>());
    }
  } else if (expr->type == NodeType::index) {
    std::string members = "{";
    bool first = true;
    for (int i = 0; i < vv_width; i++) {
      if (!first) {
        members += ",";
      }
      first = false;
      members += fmt::format("{}", expr->index_offset(i));
    }
    TC_ASSERT(bit::is_power_of_two(num_groups));
    members += "}";
    auto index_id = expr->index_id(0);
    auto base = index_name(prog->current_snode, index_id);
    emit_code("auto {}_index = {}({} + {} * 0);", expr->var_name,
              vv_type(DataType::i32), base, num_groups);
    auto constant =
        get_constant(fmt::format("{}({})", vv_type(expr->data_type), members));

    emit_code("auto {} = add({}_index, {});", expr->var_name, expr->var_name,
              constant);
  } else if (expr->type == NodeType::pointer) {
    emit_code("{} *{}[{}];", expr->data_type_name(), expr->var_name, vv_width);
    TC_WARN("Vectorized pointer of different SNodes is unsupported!");
    emit_code("for (int v = 0; v < {}; v++)", vv_width);
    std::vector<std::string> elems(max_num_indices, ", 0");
    auto snode = expr->ch[0]->new_addresses(0);
    for (int i = 1; i < (int)expr->ch.size(); i++) {
      elems[snode->index_order[i - 1]] =
          fmt::format(", {}.element(v)", expr->ch[i]->var_name);
    }
    std::string total_elem = "";
    for (int i = 0; i < max_num_indices; i++) {
      total_elem += elems[i];
    }
    emit_code("{}[v] = access_{}(context.buffers[0] {});", expr->var_name,
              expr->ch[0]->new_addresses(0)->node_type_name, total_elem);
  } else if (expr->type == NodeType::adapter_store) {
    auto &ad = prog->adapter(expr[1]->value<int>());
    /*
    emit_code("{}.store(&{}.inputs[{}]);", expr[0]->var_name,
              adapter_name(expr[1]->members[0]->value<int>()),
              expr[2]->members[0]->value<int>() / ad.input_group_size);
              */
    ad.store_exprs[expr[2]->value<int>() / ad.input_group_size].set(expr[0]);
  } else if (expr->type == NodeType::adapter_load) {
    // generate offset
    auto &ad = prog->adapter(expr[0]->value<int>());
    std::vector<int> offsets_val;
    for (int i = 0; i < num_groups; i++) {
      for (int j = 0; j < ad.output_group_size; j++) {
        int elem_id = expr[1]->value<int>();
        offsets_val.push_back(i * ad.input_group_size +
                              elem_id / ad.input_group_size *
                                  ad.input_group_size * num_groups +
                              elem_id % ad.input_group_size);
      }
    }
    auto offsets = offsets_val;
    emit_code("{} {};", vv_type(ad.dt), expr->var_name);
    int input_vv_width = ad.input_group_size * num_groups;
    for (int i = 0; i < split; i++) {
      // For each
      std::vector<int> offset_subset(offsets.begin() + i * simd_width,
                                     offsets.begin() + (i + 1) * simd_width);

      std::vector<int> register_id(simd_width);
      std::vector<int> register_offset(simd_width);

      for (int j = 0; j < simd_width; j++) {
        register_id[j] = offset_subset[j] / simd_width;
        register_offset[j] = offset_subset[j] % simd_width;
      }

      auto sorted = register_id;
      std::sort(std::begin(sorted), std::end(sorted));
      sorted.resize(std::unique(sorted.begin(), sorted.end()) - sorted.begin());

      for (int k = 0; k < (int)sorted.size(); k++) {
        auto rid = sorted[k];
        auto tmp_arg = vec_to_list_tmp(register_offset);
        int mask = 0;
        for (int j = 0; j < simd_width; j++) {
          if (register_id[j] == rid) {
            mask += 1 << j;
          }
        }
        auto src = fmt::format(
            "{}.d[{}]",
            ad.store_exprs[rid / (input_vv_width / simd_width)]->var_name,
            rid % (input_vv_width / simd_width));
        auto v = fmt::format("{}.d[{}]", expr->var_name, i);
        auto shuffled = fmt::format("shuffle8x32{}({})", tmp_arg, src);
        if (k == 0) {
          emit_code("{} = {};", v, shuffled);
        } else {
          emit_code("{} = blend({}, {}, {});", v, v, shuffled, mask);
        }
      }
    }
  } else {
    TC_ERROR("Node {} cannot be visited.", expr->node_type_name());
  }
}

TLANG_NAMESPACE_END
