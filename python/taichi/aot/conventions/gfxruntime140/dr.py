"""
Data representation of all JSON data structures following the GfxRuntime140
convention.
"""
from typing import Any, Dict, List, Optional

from taichi.aot.utils import dump_json_data_model, json_data_model


@json_data_model
class FieldAttributes:
    def __init__(self, j: Dict[str, Any]) -> None:
        dtype = j["dtype"]
        dtype_name = j["dtype_name"]
        element_shape = j["element_shape"]
        field_name = j["field_name"]
        is_scalar = j["is_scalar"]
        mem_offset_in_parent = j["mem_offset_in_parent"]
        shape = j["shape"]

        self.dtype: int = int(dtype)
        self.dtype_name: str = str(dtype_name)
        self.element_shape: List[int] = [int(x) for x in element_shape]
        self.field_name: str = str(field_name)
        self.is_scalar: bool = bool(is_scalar)
        self.mem_offset_in_parent: int = int(mem_offset_in_parent)
        self.shape: List[int] = [int(x) for x in shape]


@json_data_model
class ArgumentAttributes:
    def __init__(self, j: Dict[str, Any]) -> None:
        index = j["key"][0]
        dtype = j["value"]["dtype"]
        element_shape = j["value"]["element_shape"]
        field_dim = j["value"]["field_dim"]
        fmt = j["value"]["format"]
        is_array = j["value"]["is_array"]
        offset_in_mem = j["value"]["offset_in_mem"]
        stride = j["value"]["stride"]
        # (penguinliong) Note that the name field is optional for kernels.
        # Kernels are always launched by indexed arguments and this is for
        # debugging and header generation only.
        name = j["value"]["name"] if "name" in j["value"] and len(j["value"]["name"]) > 0 else None
        ptype = j["value"]["ptype"] if "ptype" in j["value"] else None

        self.dtype: int = int(dtype)
        self.element_shape: List[int] = [int(x) for x in element_shape]
        self.field_dim: int = int(field_dim)
        self.format: int = int(fmt)
        self.index: int = int(index)
        self.is_array: bool = bool(is_array)
        self.offset_in_mem: int = int(offset_in_mem)
        self.stride: int = int(stride)
        self.name: Optional[str] = str(name) if name is not None else None
        self.ptype: Optional[int] = int(ptype) if ptype is not None else None


@json_data_model
class ContextAttributes:
    def __init__(self, j: Dict[str, Any]) -> None:
        arg_attribs_vec_ = j["arg_attribs_vec_"]
        args_bytes_ = j["args_bytes_"]
        arr_access = j["arr_access"]
        ret_attribs_vec_ = j["ret_attribs_vec_"]
        rets_bytes_ = j["rets_bytes_"]

        self.arg_attribs_vec_: List[ArgumentAttributes] = [ArgumentAttributes(x) for x in arg_attribs_vec_]
        self.arg_attribs_vec_.sort(key=lambda x: x.index)
        self.args_bytes_: int = int(args_bytes_)
        self.arr_access: List[int] = [int(x["value"]) for x in arr_access]
        self.ret_attribs_vec_: List[ArgumentAttributes] = [ArgumentAttributes(x) for x in ret_attribs_vec_]
        self.rets_bytes_: int = int(rets_bytes_)


@json_data_model
class Buffer:
    def __init__(self, j: Dict[str, Any]) -> None:
        root_id = j["root_id"][0]
        ty = j["type"]

        self.root_id: int = int(root_id)
        self.type: int = int(ty)


@json_data_model
class BufferBinding:
    def __init__(self, j: Dict[str, Any]) -> None:
        binding = j["binding"]
        buffer = j["buffer"]

        self.binding: int = int(binding)
        self.buffer: Buffer = Buffer(buffer)


@json_data_model
class TextureBinding:
    def __init__(self, j: Dict[str, Any]) -> None:
        arg_id = j["arg_id"]
        binding = j["binding"]
        is_storage = j["is_storage"]

        self.arg_id: int = int(arg_id)
        self.binding: int = int(binding)
        self.is_storage: bool = bool(is_storage)


@json_data_model
class RangeForAttributes:
    def __init__(self, j: Dict[str, Any]) -> None:
        begin = j["begin"]
        const_begin = j["const_begin"]
        const_end = j["const_end"]
        end = j["end"]

        self.begin: int = int(begin)
        self.const_begin: bool = bool(const_begin)
        self.const_end: bool = bool(const_end)
        self.end: int = int(end)


@json_data_model
class TaskAttributes:
    def __init__(self, j: Dict[str, Any]) -> None:
        advisory_num_threads_per_group = j["advisory_num_threads_per_group"]
        advisory_total_num_threads = j["advisory_total_num_threads"]
        buffer_binds = j["buffer_binds"]
        name = j["name"]
        range_for_attribs = j["range_for_attribs"] if "range_for_attribs" in j else None
        task_type = j["task_type"]
        texture_binds = j["texture_binds"]

        self.advisory_num_threads_per_group: int = int(advisory_num_threads_per_group)
        self.advisory_total_num_threads: int = int(advisory_total_num_threads)
        self.buffer_binds: List[BufferBinding] = [BufferBinding(x) for x in buffer_binds]
        self.name: str = str(name)
        self.range_for_attribs: Optional[RangeForAttributes] = (
            RangeForAttributes(range_for_attribs) if range_for_attribs is not None else None
        )
        self.task_type: int = int(task_type)
        self.texture_binds: List[TextureBinding] = [TextureBinding(x) for x in texture_binds]


@json_data_model
class DeviceCapabilityLevel:
    def __init__(self, j: Dict[str, Any]) -> None:
        key = j["key"]
        value = j["value"]

        self.key: str = str(key)
        self.value: int = int(value)


@json_data_model
class KernelAttributes:
    def __init__(self, j: Dict[str, Any]) -> None:
        ctx_attribs = j["ctx_attribs"]
        is_jit_evaluator = j["is_jit_evaluator"]
        name = j["name"]
        tasks_attribs = j["tasks_attribs"]

        self.ctx_attribs: ContextAttributes = ContextAttributes(ctx_attribs)
        self.is_jit_evaluator: bool = bool(is_jit_evaluator)
        self.name: str = str(name)
        self.tasks_attribs: List[TaskAttributes] = [TaskAttributes(x) for x in tasks_attribs]


@json_data_model
class Metadata:
    def __init__(self, j: Dict[str, Any]) -> None:
        fields = j["fields"]
        kernels = j["kernels"]
        required_caps = j["required_caps"]
        root_buffer_size = j["root_buffer_size"]

        self.fields: List[FieldAttributes] = [FieldAttributes(x) for x in fields]
        self.kernels: List[KernelAttributes] = [KernelAttributes(x) for x in kernels]
        self.required_caps: List[DeviceCapabilityLevel] = [DeviceCapabilityLevel(x) for x in required_caps]
        self.root_buffer_size: int = int(root_buffer_size)


def from_json_metadata(j: Dict[str, Any]) -> Metadata:
    return Metadata(j)


def to_json_metadata(meta_data: Metadata) -> Dict[str, Any]:
    return dump_json_data_model(meta_data)


@json_data_model
class SymbolicArgument:
    def __init__(self, j: Dict[str, Any]) -> None:
        dtype_id = j["dtype_id"]
        element_shape = j["element_shape"]
        field_dim = j["field_dim"]
        name = j["name"]
        num_channels = j["num_channels"]
        tag = j["tag"]

        self.dtype_id: int = int(dtype_id)
        self.element_shape: List[int] = [int(x) for x in element_shape]
        self.field_dim: int = int(field_dim)
        self.name: str = str(name)
        self.num_channels: int = int(num_channels)
        self.tag: int = int(tag)


@json_data_model
class Dispatch:
    def __init__(self, j: Dict[str, Any]) -> None:
        kernel_name = j["kernel_name"]
        symbolic_args = j["symbolic_args"]

        self.kernel_name: str = str(kernel_name)
        self.symbolic_args: List[SymbolicArgument] = [SymbolicArgument(x) for x in symbolic_args]


@json_data_model
class GraphData:
    def __init__(self, j: Dict[str, Any]) -> None:
        dispatches = j["dispatches"]

        self.dispatches = [Dispatch(x) for x in dispatches]


@json_data_model
class Graph:
    def __init__(self, j: Dict[str, Any]) -> None:
        key = j["key"]
        value = j["value"]

        self.key = str(key)
        self.value = GraphData(value)


def from_json_graph(j: Dict[str, Any]) -> Graph:
    return Graph(j)


def to_json_graph(graph: Graph) -> Dict[str, Any]:
    return dump_json_data_model(graph)
