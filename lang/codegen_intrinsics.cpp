#include "codegen.h"
#include "program.h"

namespace taichi::Tlang {

void CPUCodeGen::visit_intrinsics(Expr &expr) {
  // TC_P(expr->id);
  // TC_P(expr->node_type_name());
  auto vv_width = num_groups * expr->group_size();
  TC_ASSERT(vv_width % simd_width == 0);
  int split = vv_width / simd_width;
  auto vv_type = [&](DataType dt) {
    return fmt::format("vvec<{}, {}, {}>", data_type_name(dt), simd_width,
                       split);
  };
  auto vvec_const_str = [&](DataType dt, auto val) {
    return fmt::format("{}({})", vv_type(dt), val);
  };
  auto vvec_const_str_list = [&](DataType dt, auto val) {
    return fmt::format("{}({})", vv_type(dt), vec_to_list_str(val));
  };
  TC_ASSERT(expr->is_vectorized);
  TC_ASSERT(expr->members.size() == 0 ||
            (int)expr->members.size() == group_size);
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

  if (binary_ops_intrinsics.find(expr->type) != binary_ops_intrinsics.end()) {
    auto op = binary_ops_intrinsics[expr->type];
    emit_code("auto {} = {}({}, {});", expr->var_name, op,
              expr->ch[0]->var_name, expr->ch[1]->var_name);
    // emit_code("{}.print();", expr->var_name);
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
  } else if (expr->type == NodeType::load) {
    bool regular = false;
    if (expr[0]->type == NodeType::pointer &&
        expr[0][1]->type == NodeType::index) {
      regular = true;
      TC_INFO("regular");
    } else {
      TC_INFO("irregular");
    }
    if (regular) {
      // TODO: irregular case
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
      emit_code("vvec<{}, {}, {}> {}({});", expr->data_type_name(), simd_width,
                split, expr->var_name, get_vectorized_address(addr, 0, 0));
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
      emit_code("auto {} = {}({}_base, {}_offsets);", expr->var_name,
                vv_type(expr->data_type), expr[0]->var_name, expr[0]->var_name);
    }
  } else if (expr->type == NodeType::store) {
    bool regular = true;
    if (regular) {
      // TODO: analyze address here
      auto addr = expr[0][0]->get_address();
      TC_ASSERT(addr.coeff_aosoa_group_size == 0 ||
                num_groups % addr.coeff_aosoa_group_size == 0);
      emit_code("{}.store({});", expr->ch[1]->var_name,
                get_vectorized_address(addr, 0, 0));
    } else {
      // AVX2 has no scatter...
      emit_code("{}.store({}_base, {}_offsets);", expr->ch[1]->var_name,
                expr[0]->var_name, expr[0]->var_name);
    }
  } else if (expr->type == NodeType::combine) {
    // do nothing
  } else if (expr->type == NodeType::imm) {
    TC_WARN("Using member imm");
    if (expr->data_type == DataType::i32) {
      emit_code("auto {} = vvec<int32, {}, {}>({}); /*i32*/ ", expr->var_name,
                simd_width, split, (int64)expr->members[0]->value<int32>());
    } else {
      emit_code("auto {} = vvec<float32, {}, {}>({}); /*i32*/ ", expr->var_name,
                simd_width, split, (int64)expr->members[0]->value<float32>());
    }
  } else if (expr->type == NodeType::index) {
    std::string members = "{";
    bool first = true;
    for (int i = 0; i < num_groups; i++) {
      for (int j = 0; j < expr->group_size(); j++) {
        if (!first) {
          members += ",";
        }
        first = false;
        members += fmt::format("b * {} + {}", num_groups, i);
      }
    }
    members += "}";
    emit_code("auto {} = {}({});", expr->var_name, vv_type(expr->data_type),
              members);
  } else if (expr->type == NodeType::pointer) {
    // emit base pointer and offsets
    auto addr = expr[0]->get_address_();
    auto buffer_name = fmt::format("context.buffers[{:02d}]", addr.buffer_id);
    emit_code("auto *{}_base = ({} *){} + {} * n;", expr->var_name,
              data_type_name(expr->data_type), buffer_name, addr.coeff_imax);

    auto index = expr->ch[1]->var_name;

    std::vector<int> coeff_const;
    for (int i = 0; i < num_groups; i++) {
      for (auto &m : expr->ch[0]->members) {
        coeff_const.push_back(m->get_address_().coeff_const);
      }
    }
    auto offset_var = vvec_const_str_list(DataType::i32, coeff_const);
    if (addr.coeff_aosoa_stride != 0) {
      emit_code("auto {}_offsets = {} + {} * {} + {} / {} * {};",
                expr->var_name, offset_var,
                vvec_const_str(DataType::i32, addr.coeff_i), index, index,
                vvec_const_str(DataType::i32, addr.coeff_aosoa_group_size),
                vvec_const_str(DataType::i32, addr.coeff_aosoa_stride));
    } else {
      emit_code("auto {}_offsets = {} + {} * {};", expr->var_name, offset_var,
                vvec_const_str(DataType::i32, addr.coeff_i), index);
    }
  } else if (expr->type == NodeType::adapter_store) {
    auto &ad = prog->adapter(expr[1]->members[0]->value<int>());
    TC_P(ad.input_group_size);
    TC_P(expr[2]->members[0]->value<int>());
    emit_code("{}.set<{}>({});",
              adapter_name(expr[1]->members[0]->value<int>()),
              expr[2]->members[0]->value<int>() / ad.input_group_size,
              expr[0]->var_name);
  } else if (expr->type == NodeType::adapter_load) {
    // generate offset
    TC_P(num_groups);
    auto &ad = prog->adapter(expr[0]->members[0]->value<int>());
    std::vector<int> offsets_val;
    for (int i = 0; i < num_groups; i++) {
      for (int j = 0; j < ad.output_group_size; j++) {
        int elem_id = expr[1]->members[j]->value<int>();
        offsets_val.push_back(i * ad.input_group_size +
                              elem_id / ad.input_group_size *
                                  ad.input_group_size * num_groups +
                              elem_id % ad.input_group_size);
      }
    }
    auto offsets = vvec_const_str_list(DataType::i32, offsets_val);
    emit_code("auto {} = shuffle({}, {});", expr->var_name,
              adapter_name(expr[0]->members[0]->value<int>()), offsets);
    // emit_code("{}.print();", expr->var_name);
  } else {
    TC_ERROR("Node {} cannot be visited.", expr->node_type_name());
  }
}
}
