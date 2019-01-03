#include "codegen_cpu.h"
#include "program.h"

namespace taichi::Tlang {

void CPUCodeGen::visit_vv(Expr &expr) {
  // TC_P(expr->id);
  // TC_P(expr->node_type_name());
  auto vv_width = num_groups * expr->group_size();
  TC_ASSERT(expr->is_vectorized);
  TC_ASSERT(expr->members.size() == 0 ||
            (int)expr->members.size() == group_size);
  if (expr->type == NodeType::addr) {
    return;
  }
  if (expr->var_name == "") {
    expr->var_name = create_variable();
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
    emit_code("auto {} = {} {} {};", expr->var_name, expr->ch[0]->var_name, op,
              expr->ch[1]->var_name);
    // emit_code("{}.print();", expr->var_name);
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
  } else if (expr->type == NodeType::land) {
    emit_code("auto {} = land({}, {});", expr->var_name, expr[0]->var_name,
              expr[1]->members[0]->value<int>());
  } else if (expr->type == NodeType::shr) {
    emit_code("auto {} = shr({}, {});", expr->var_name, expr[0]->var_name,
              expr[1]->members[0]->value<int>());
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
  } else if (expr->type == NodeType::load) {
    emit_code("auto {} = load<{}, {}>({}_base, {}_offsets);", expr->var_name,
              expr->group_size() * num_groups, data_type_name(expr->data_type),
              expr[0]->var_name, expr[0]->var_name);
  } else if (expr->type == NodeType::store) {
    emit_code("store({}, {}_base, {}_offsets);", expr->ch[1]->var_name,
              expr->ch[0]->var_name, expr->ch[0]->var_name);
  } else if (expr->type == NodeType::combine) {
    // do nothing
  } else if (expr->type == NodeType::imm) {
    TC_WARN("Using member imm");
    if (expr->data_type == DataType::i32) {
      emit_code("auto {} = {}; /*i32*/ ", expr->var_name,
                vv_constant_str(group_size * num_groups, DataType::i32,
                                (int64)expr->members[0]->value<int32>()));
    } else {
      emit_code("auto {} = {}; /*f32*/ ", expr->var_name,
                vv_constant_str(group_size * num_groups, DataType::f32,
                                expr->members[0]->value<float32>()));
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
    emit_code("auto {} = {}({});", expr->var_name,
              vv_type_str(num_groups * expr->group_size(), DataType::i32),
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
    auto offset_var = vv_constant_str(vv_width, DataType::i32, coeff_const);
    if (addr.coeff_aosoa_stride != 0) {
      emit_code(
          "auto {}_offsets = {} + {} * {} + {} / {} * {};", expr->var_name,
          offset_var, vv_constant_str(vv_width, DataType::i32, addr.coeff_i),
          index, index,
          vv_constant_str(vv_width, DataType::i32, addr.coeff_aosoa_group_size),
          vv_constant_str(vv_width, DataType::i32, addr.coeff_aosoa_stride));
    } else {
      emit_code("auto {}_offsets = {} + {} * {};", expr->var_name, offset_var,
                vv_constant_str(num_groups * expr->group_size(), DataType::i32,
                                addr.coeff_i),
                index);
    }
  } else if (expr->type == NodeType::adapter_store) {
    // Do nothing
    // create_adapter(DataType::f32, 0, 1, 8);
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
    auto offsets = vv_constant_str(ad.output_group_size * num_groups,
                                   DataType::i32, offsets_val);
    emit_code("auto {} = shuffle({}, {});", expr->var_name,
              adapter_name(expr[0]->members[0]->value<int>()), offsets);
    // emit_code("{}.print();", expr->var_name);
  } else {
    TC_ERROR("Node {} cannot be visited.", expr->node_type_name());
  }
}  // namespace taichi::Tlang
}  // namespace taichi::Tlang
