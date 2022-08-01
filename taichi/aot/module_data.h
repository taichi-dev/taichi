#pragma once

#include <string>
#include <vector>

#include "taichi/common/core.h"
#include "taichi/common/serialization.h"

namespace taichi {
namespace lang {
namespace aot {

struct CompiledFieldData {
  std::string field_name;
  uint32_t dtype{0};
  std::string dtype_name;
  size_t mem_offset_in_parent{0};
  std::vector<int> shape;
  bool is_scalar{false};
  std::vector<int> element_shape;

  TI_IO_DEF(field_name,
            dtype,
            dtype_name,
            mem_offset_in_parent,
            shape,
            is_scalar,
            element_shape);
};

enum class BufferType { Root, GlobalTmps, Args, Rets };

struct BufferInfo {
  BufferType type;
  int id{-1};  // only used if type==Root

  TI_IO_DEF(type, id);
};

struct BufferBind {
  BufferInfo buffer;
  int binding{0};

  TI_IO_DEF(buffer, binding);
};

struct TextureBind {
  int arg_id;
  int binding;
  bool is_storage;

  TI_IO_DEF(arg_id, binding, is_storage);
};

struct CompiledOffloadedTask {
  std::string type;
  std::string range_hint;
  std::string name;
  // Do we need to inline the source code?
  std::string source_path;
  int gpu_block_size{0};

  std::vector<BufferBind> buffer_binds;
  std::vector<TextureBind> texture_binds;

  TI_IO_DEF(type,
            range_hint,
            name,
            source_path,
            gpu_block_size,
            buffer_binds,
            texture_binds);
};

struct ScalarArg {
  std::string dtype_name;
  // Unit: byte
  size_t offset_in_args_buf{0};

  TI_IO_DEF(dtype_name, offset_in_args_buf);
};

struct ArrayArg {
  std::string dtype_name;
  std::size_t field_dim{0};
  // If |element_shape| is empty, it means this is a scalar
  std::vector<int> element_shape;
  // Unit: byte
  std::size_t shape_offset_in_args_buf{0};
  // For Vulkan/OpenGL/Metal, this is the binding index
  int bind_index{0};

  TI_IO_DEF(dtype_name,
            field_dim,
            element_shape,
            shape_offset_in_args_buf,
            bind_index);
};

struct CompiledTaichiKernel {
  std::vector<CompiledOffloadedTask> tasks;
  int args_count{0};
  int rets_count{0};
  size_t args_buffer_size{0};
  size_t rets_buffer_size{0};

  std::unordered_map<int, ScalarArg> scalar_args;
  std::unordered_map<int, ArrayArg> arr_args;

  TI_IO_DEF(tasks,
            args_count,
            rets_count,
            args_buffer_size,
            rets_buffer_size,
            scalar_args,
            arr_args);
};

struct ModuleData {
  std::unordered_map<std::string, CompiledTaichiKernel> kernels;
  std::unordered_map<std::string, CompiledTaichiKernel> kernel_tmpls;
  std::vector<aot::CompiledFieldData> fields;

  size_t root_buffer_size;

  void dump_json(std::string path) {
    TextSerializer ts;
    ts.serialize_to_json("aot_data", *this);
    ts.write_to_file(path);
  }

  TI_IO_DEF(kernels, kernel_tmpls, fields, root_buffer_size);
};

}  // namespace aot
}  // namespace lang
}  // namespace taichi
