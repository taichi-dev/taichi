#include "codegen_cpu.h"
#include "program.h"
#include <taichi/common/bit.h>

TLANG_NAMESPACE_BEGIN

void CPUCodeGen::visit_intrinsics(Expr &expr) {
  /*
  TC_P(expr->id);
  TC_P(expr->node_type_name());
  TC_P(num_groups);
  TC_P(expr->lanes);
  */
  auto vv_width = expr->lanes;
  TC_ASSERT(vv_width == 1 || vv_width == simd_width);
  auto vec_type = [&](DataType dt) {
    if (expr->lanes == 1) {
      return fmt::format("vec<{}, {}>", data_type_name(dt), 1);
    } else {
      return fmt::format("vec<{}, {}>", data_type_name(dt), simd_width);
    }
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
  }
  if (visited.find(expr) != visited.end()) {
    return;
  }
  visited.insert(expr);

  for (auto &m : expr->members) {
    if (!m->name().empty()) {
      emit_code("// @ {}", m->name());
    }
  }

  auto address_elements = [&](SNode *snode, std::string index, int start = 1,
                              bool last_zero = false) {
    std::vector<std::string> elems(max_num_indices, ", 0");
    for (int j = start; j < (int)expr->ch.size() - (int)last_zero; j++) {
      elems[snode->index_order[j - start]] =
          fmt::format(", {}.element({})", expr->ch[j]->var_name, index);
    }
    std::string total_elem = "";
    for (int j = 0; j < max_num_indices; j++) {
      total_elem += elems[j];
    }
    return total_elem;
  };

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
      emit_code("auto {} = cast<int32>({});", expr->var_name,
                expr[0]->var_name);
    } else if (expr->data_type == DataType::f32) {
      emit_code("auto {} = cast<float32>({});", expr->var_name,
                expr[0]->var_name);
    } else {
      TC_NOT_IMPLEMENTED
    }
  } else if (expr->type == NodeType::vload) {
    emit_code("auto {} = {}::load(access_{}(context.buffers[0] {}));",
              expr->var_name, vec_type(expr->data_type),
              expr[0]->snode_ptr(0)->node_type_name,
              address_elements(expr[0]->snode_ptr(0), "0"));
  } else if (expr->type == NodeType::vstore) {
    emit_code("{}.store(access_{}(context.buffers[0] {}));", expr[1]->var_name,
              expr[0]->snode_ptr(0)->node_type_name,
              address_elements(expr[0]->snode_ptr(0), "0", 2));
  } else if (expr->type == NodeType::gather) {
    // gather offsets
    emit_code("auto {}_offsets = {};", expr->var_name, expr->ch.back()->var_name);
    emit_code(
        "auto {} = gather<{}, {}>(access_{}(context.buffers[0] {}), {}_offsets);",
        expr->var_name, expr->data_type_name(), simd_width,
        expr[0]->snode_ptr(0)->node_type_name,
        address_elements(expr[0]->snode_ptr(0), "0", 1, true), expr->var_name);
  } else if (expr->type == NodeType::load) {
    emit_code("auto {} = {}::load({});", expr->var_name,
              vec_type(expr->data_type), expr[0]->var_name);
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
                vec_type(expr->data_type), expr[0]->var_name,
                expr[0]->var_name);
    }
#endif
  } else if (expr->type == NodeType::store) {
    emit_code("{}.store({});", expr->ch[1]->var_name, expr[0]->var_name);
  } else if (expr->type == NodeType::combine) {
    // do nothing
  } else if (expr->type == NodeType::imm) {
    if (expr->data_type == DataType::i32) {
      std::vector<int32> values;
      for (int i = 0; i < expr->lanes; i++) {
        values.push_back(expr->value<int32>(i));
      }
      auto constant = get_constant(fmt::format(
          "{}({})", vec_type(expr->data_type), vec_to_list(values, "{")));
      emit_code("auto {} = {}; /*i32*/ ", expr->var_name, constant);
    } else {
      std::vector<float32> values;
      for (int i = 0; i < expr->lanes; i++) {
        values.push_back(expr->value<float32>(i));
      }
      auto constant = get_constant(fmt::format(
          "{}({})", vec_type(expr->data_type), vec_to_list(values, "{")));
      emit_code("auto {} = {}; /*f32*/ ", expr->var_name, constant);
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
    auto snode = prog->current_snode;
    if (snode->type == SNodeType::indirect && index_id == snode->index_id) {
      // indirect node, needs an load from "pointer" array
      auto base = loop_variable(prog->current_snode);
      emit_code("auto {}_index = {}({});", expr->var_name,
                vec_type(DataType::i32), base);
      auto constant = get_constant(
          fmt::format("{}({})", vec_type(expr->data_type), members));

      emit_code("auto {}_indirect = add({}_index, {});", expr->var_name,
                expr->var_name, constant);
      bool gather = false;  // use loadu
      if (gather) {
        emit_code(
            "auto {} = gather<{}, {}>((void *)&{}_cache->data[0], "
            "{}_indirect);",
            expr->var_name, data_type_name(expr->data_type), simd_width,
            prog->current_snode->node_type_name, expr->var_name);
      } else {
        emit_code(
            "auto {} = load<{}, {}>((uint8 *)&{}_cache->data[0] + "
            "{}_indirect.element(0) * sizeof({}));",
            expr->var_name, data_type_name(expr->data_type), simd_width,
            prog->current_snode->node_type_name, expr->var_name,
            expr->data_type_name());
      }
    } else {
      auto base = index_name_global(prog->current_snode, index_id);
      emit_code("auto {}_index = {}({});", expr->var_name,
                vec_type(DataType::i32), base);
      auto constant = get_constant(
          fmt::format("{}({})", vec_type(expr->data_type), members));

      emit_code("auto {} = add({}_index, {});", expr->var_name, expr->var_name,
                constant);
    }
  } else if (expr->type == NodeType::pointer) {
    emit_code("{} *{}[{}];", expr->data_type_name(), expr->var_name, vv_width);
    for (int i = 0; i < vv_width; i++) {
      auto snode = expr._address()->snode_ptr(i);
      std::vector<std::string> elems(max_num_indices, ", 0");
      for (int j = 1; j < (int)expr->ch.size(); j++) {
        elems[snode->index_order[j - 1]] =
            fmt::format(", {}.element({})", expr->ch[j]->var_name, i);
      }
      std::string total_elem = "";
      for (int j = 0; j < max_num_indices; j++) {
        total_elem += elems[j];
      }
      emit_code("{}[{}] = access_{}(context.buffers[0] {});", expr->var_name, i,
                expr._address()->snode_ptr(i)->node_type_name, total_elem);
    }
  } else if (expr->type == NodeType::print) {
    emit_code("auto {} = {};", expr->var_name, expr->ch[0]->var_name);
    emit_code("{}.print();", expr->var_name);
  } else if (expr->type == NodeType::adapter_store) {
    // save ch[0] to adapter ch[1]
    auto &ad = current_kernel->adapter(expr[1]->value<int>());
    ad.store_exprs[expr[2]->value<int>()].set(expr[0]);
  } else if (expr->type == NodeType::adapter_load) {
    auto &ad = current_kernel->adapter(expr[0]->value<int>());
    std::vector<int> offsets_val(vv_width);

    for (int i = 0; i < vv_width; i++) {
      offsets_val[i] = expr[1]->value<int>(i);
      TC_ASSERT(expr[1]->attribute<int>(1, i));
    }

    auto offsets = offsets_val;
    emit_code("{} {};", vec_type(ad.dt), expr->var_name);

    // For each split (vec) of vvec
    std::vector<int> offset_subset(offsets.begin(),
                                   offsets.begin() + simd_width);

    std::vector<int> register_id(simd_width);
    std::vector<int> register_offset(simd_width);

    for (int j = 0; j < simd_width; j++) {
      register_id[j] = offset_subset[j] / simd_width;
      register_offset[j] = offset_subset[j] % simd_width;
    }

    auto sorted = register_id;
    std::sort(std::begin(sorted), std::end(sorted));
    sorted.resize(std::unique(sorted.begin(), sorted.end()) - sorted.begin());

    // for each unique register_id...
    for (int k = 0; k < (int)sorted.size(); k++) {
      auto rid = sorted[k];
      auto tmp_arg = vec_to_list_tmp(register_offset);
      int mask = 0;
      for (int j = 0; j < simd_width; j++) {
        if (register_id[j] == rid) {
          mask += 1 << j;
        }
      }
      auto src = fmt::format("{}", ad.store_exprs[rid]->var_name);
      auto v = fmt::format("{}", expr->var_name);
      auto shuffled = fmt::format("shuffle8x32{}({})", tmp_arg, src);
      if (k == 0) {
        emit_code("{} = {};", v, shuffled);
      } else {
        emit_code("{} = blend({}, {}, {});", v, v, shuffled, mask);
      }
    }
  } else if (expr->type == NodeType::touch) {
    for (int i = 0; i < simd_width; i++) {
      // val comes first, then indices
      emit_code("touch_{}(context.buffers[0], {}.element({}), {}.element({}));",
                expr->snode_ptr(i)->node_type_name, expr->ch[1]->var_name, i,
                expr->ch[0]->var_name, i);
    }
  } else if (expr->type == NodeType::reduce) {
    TC_INFO("Reduce optimization");
    // TC_ASSERT(expr[1]->type == NodeType::pointer && expr[1])
    // for (int i = 0; i < simd_width; i++) {
    // val comes first, then indices
    // emit_code("*{}[{}] += {}.element({});", expr[0]->var_name, i,
    // expr->ch[1]->var_name, i);
    emit_code("sum = add(sum, {});", expr->ch[1]->var_name);
    {
      CODE_REGION(interior_shared_variable_begin);
      emit_code("vec<{}, {}> sum(0);", expr[1]->data_type_name(), simd_width);
    }
    {
      CODE_REGION(interior_shared_variable_end);
      auto snode = expr[0][0]->snode_ptr(0);
      std::vector<std::string> elems(max_num_indices, ", 0");
      for (int j = 0; j < (int)max_num_indices; j++) {
        elems[j] = fmt::format(", index_{}_{}_global",
                               prog->current_snode->parent->node_type_name, j);
      }
      std::string total_elem = "";
      for (int j = 0; j < max_num_indices; j++) {
        total_elem += elems[j];
      }
      emit_code("auto *reduce_target = access_{}(context.buffers[0] {});",
                snode->node_type_name, total_elem);
      emit_code("*{} += reduce_sum({});", "reduce_target", "sum");
      // emit_code("std::cout << reduce_sum(sum) << std::endl;");
    }
    //}
  } else {
    TC_ERROR("Node {} cannot be visited.", expr->node_type_name());
  }
}

TLANG_NAMESPACE_END
