from taichi._lib import core as _ti_core

Layout = _ti_core.Layout
AutodiffMode = _ti_core.AutodiffMode
SNodeGradType = _ti_core.SNodeGradType
Format = _ti_core.Format


class DeviceCapability:
    spirv_version_1_3 = "spirv_version=1.3"
    spirv_version_1_4 = "spirv_version=1.4"
    spirv_version_1_5 = "spirv_version=1.5"
    spirv_has_int8 = "spirv_has_int8"
    spirv_has_int16 = "spirv_has_int16"
    spirv_has_int64 = "spirv_has_int64"
    spirv_has_float16 = "spirv_has_float16"
    spirv_has_float64 = "spirv_has_float64"
    spirv_has_atomic_i64 = "spirv_has_atomic_i64"
    spirv_has_atomic_float16 = "spirv_has_atomic_float16"
    spirv_has_atomic_float16_add = "spirv_has_atomic_float16_add"
    spirv_has_atomic_float16_minmax = "spirv_has_atomic_float16_minmax"
    spirv_has_atomic_float = "spirv_has_atomic_float"
    spirv_has_atomic_float_add = "spirv_has_atomic_float_add"
    spirv_has_atomic_float_minmax = "spirv_has_atomic_float_minmax"
    spirv_has_atomic_float64 = "spirv_has_atomic_float64"
    spirv_has_atomic_float64_add = "spirv_has_atomic_float64_add"
    spirv_has_atomic_float64_minmax = "spirv_has_atomic_float64_minmax"
    spirv_has_variable_ptr = "spirv_has_variable_ptr"
    spirv_has_physical_storage_buffer = "spirv_has_physical_storage_buffer"
    spirv_has_subgroup_basic = "spirv_has_subgroup_basic"
    spirv_has_subgroup_vote = "spirv_has_subgroup_vote"
    spirv_has_subgroup_arithmetic = "spirv_has_subgroup_arithmetic"
    spirv_has_subgroup_ballot = "spirv_has_subgroup_ballot"
    spirv_has_non_semantic_info = "spirv_has_non_semantic_info"
    spirv_has_no_integer_wrap_decoration = "spirv_has_no_integer_wrap_decoration"


__all__ = [
    'Layout', 'AutodiffMode', 'SNodeGradType', 'Format', 'DeviceCapability'
]
