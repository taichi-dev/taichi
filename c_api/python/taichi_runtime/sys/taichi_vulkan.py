"""
# Vulkan Backend Features

Taichi's Vulkan API gives you further control over the Vulkan version and extension requirements and allows you to interop with external Vulkan applications with shared resources.

"""
import ctypes
from ._lib import _LIB
from .taichi_core import *


"""
Structure `TiVulkanRuntimeInteropInfo` (1.4.0)

Necessary detail to share the same Vulkan runtime between Taichi and external procedures.


**NOTE** `compute_queue` and `graphics_queue` can be the same if the queue family have `VK_QUEUE_COMPUTE_BIT` and `VK_QUEUE_GRAPHICS_BIT` set at the same tiem.
"""


class TiVulkanRuntimeInteropInfo(ctypes.Structure):
    pass


TiVulkanRuntimeInteropInfo._fields_ = [
    # Pointer to Vulkan loader function `vkGetInstanceProcAddr`.
    ("get_instance_proc_addr", PFN_vkGetInstanceProcAddr),
    # Target Vulkan API version.
    ("api_version", ctypes.c_uint32),
    # Vulkan instance handle.
    ("instance", VkInstance),
    # Vulkan physical device handle.
    ("physical_device", VkPhysicalDevice),
    # Vulkan logical device handle.
    ("device", VkDevice),
    # Vulkan queue handle created in the queue family at `structure.vulkan_runtime_interop_info.compute_queue_family_index`.
    ("compute_queue", VkQueue),
    # Index of a Vulkan queue family with the `VK_QUEUE_COMPUTE_BIT` set.
    ("compute_queue_family_index", ctypes.c_uint32),
    # Vulkan queue handle created in the queue family at `structure.vulkan_runtime_interop_info.graphics_queue_family_index`.
    ("graphics_queue", VkQueue),
    # Index of a Vulkan queue family with the `VK_QUEUE_GRAPHICS_BIT` set.
    ("graphics_queue_family_index", ctypes.c_uint32),
]


"""
Structure `TiVulkanMemoryInteropInfo` (1.4.0)

Necessary detail to share the same piece of Vulkan buffer between Taichi and external procedures.
"""


class TiVulkanMemoryInteropInfo(ctypes.Structure):
    pass


TiVulkanMemoryInteropInfo._fields_ = [
    # Vulkan buffer.
    ("buffer", VkBuffer),
    # Size of the piece of memory in bytes.
    ("size", ctypes.c_uint64),
    # Vulkan buffer usage. In most of the cases, Taichi requires the `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`.
    ("usage", VkBufferUsageFlags),
    # Device memory binded to the Vulkan buffer.
    ("memory", VkDeviceMemory),
    # Offset in `VkDeviceMemory` object to the beginning of this allocation, in bytes.
    ("offset", ctypes.c_uint64),
]


"""
Structure `TiVulkanImageInteropInfo` (1.4.0)

Necessary detail to share the same piece of Vulkan image between Taichi and external procedures.
"""


class TiVulkanImageInteropInfo(ctypes.Structure):
    pass


TiVulkanImageInteropInfo._fields_ = [
    # Vulkan image.
    ("image", VkImage),
    # Vulkan image allocation type.
    ("image_type", VkImageType),
    # Pixel format.
    ("format", VkFormat),
    # Image extent.
    ("extent", VkExtent3D),
    # Number of mip-levels of the image.
    ("mip_level_count", ctypes.c_uint32),
    # Number of array layers.
    ("array_layer_count", ctypes.c_uint32),
    # Number of samples per pixel.
    ("sample_count", VkSampleCountFlagBits),
    # Image tiling.
    ("tiling", VkImageTiling),
    # Vulkan image usage. In most cases, Taichi requires the `VK_IMAGE_USAGE_STORAGE_BIT` and the `VK_IMAGE_USAGE_SAMPLED_BIT`.
    ("usage", VkImageUsageFlags),
]


_LIB.ti_create_vulkan_runtime_ext.argtypes = [
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_void_p,
]
_LIB.ti_create_vulkan_runtime_ext.restype = TiRuntime


def ti_create_vulkan_runtime_ext(
    api_version: ctypes.c_uint32,
    instance_extension_count: ctypes.c_uint32,
    instance_extensions: ctypes.c_void_p,
    device_extension_count: ctypes.c_uint32,
    device_extensions: ctypes.c_void_p,
) -> TiRuntime:
    """
    Function `ti_create_vulkan_runtime_ext` (1.4.0)

    Creates a Vulkan Taichi runtime with user-controlled capability settings.

    Return value: TiRuntime

    Parameters:
        api_version (`ctypes.c_uint32`):
        instance_extension_count (`ctypes.c_uint32`):
        instance_extensions (`ctypes.c_void_p`):
        device_extension_count (`ctypes.c_uint32`):
        device_extensions (`ctypes.c_void_p`):
    """
    out = _LIB.ti_create_vulkan_runtime_ext(
        api_version, instance_extension_count, instance_extensions, device_extension_count, device_extensions
    )
    return TiRuntime(out)


_LIB.ti_import_vulkan_runtime.argtypes = [
    ctypes.c_void_p,  # const TiVulkanRuntimeInteropInfo*,
]
_LIB.ti_import_vulkan_runtime.restype = TiRuntime


def ti_import_vulkan_runtime(
    interop_info: ctypes.c_void_p,  # const TiVulkanRuntimeInteropInfo*
) -> TiRuntime:
    """
    Function `ti_import_vulkan_runtime` (1.4.0)

    Imports the Vulkan runtime owned by Taichi to external procedures.

    Return value: TiRuntime

    Parameters:
        interop_info (`TiVulkanRuntimeInteropInfo`):
    """
    out = _LIB.ti_import_vulkan_runtime(interop_info)
    return TiRuntime(out)


_LIB.ti_export_vulkan_runtime.argtypes = [
    TiRuntime,
    ctypes.c_void_p,  # TiVulkanRuntimeInteropInfo*,
]
_LIB.ti_export_vulkan_runtime.restype = None


def ti_export_vulkan_runtime(
    runtime: TiRuntime,
    interop_info: ctypes.c_void_p,  # TiVulkanRuntimeInteropInfo*
) -> None:
    """
    Function `ti_export_vulkan_runtime` (1.4.0)

    Exports a Vulkan runtime from external procedures to Taichi.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiVulkanRuntimeInteropInfo`):
    """
    out = _LIB.ti_export_vulkan_runtime(runtime, interop_info)


_LIB.ti_import_vulkan_memory.argtypes = [
    TiRuntime,
    ctypes.c_void_p,  # const TiVulkanMemoryInteropInfo*,
]
_LIB.ti_import_vulkan_memory.restype = TiMemory


def ti_import_vulkan_memory(
    runtime: TiRuntime,
    interop_info: ctypes.c_void_p,  # const TiVulkanMemoryInteropInfo*
) -> TiMemory:
    """
    Function `ti_import_vulkan_memory` (1.4.0)

    Imports the Vulkan buffer owned by Taichi to external procedures.

    Return value: TiMemory

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiVulkanMemoryInteropInfo`):
    """
    out = _LIB.ti_import_vulkan_memory(runtime, interop_info)
    return TiMemory(out)


_LIB.ti_export_vulkan_memory.argtypes = [
    TiRuntime,
    TiMemory,
    ctypes.c_void_p,  # TiVulkanMemoryInteropInfo*,
]
_LIB.ti_export_vulkan_memory.restype = None


def ti_export_vulkan_memory(
    runtime: TiRuntime,
    memory: TiMemory,
    interop_info: ctypes.c_void_p,  # TiVulkanMemoryInteropInfo*
) -> None:
    """
    Function `ti_export_vulkan_memory` (1.4.0)

    Exports a Vulkan buffer from external procedures to Taichi.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        memory (`TiMemory`):
        interop_info (`TiVulkanMemoryInteropInfo`):
    """
    out = _LIB.ti_export_vulkan_memory(runtime, memory, interop_info)


_LIB.ti_import_vulkan_image.argtypes = [
    TiRuntime,
    ctypes.c_void_p,  # const TiVulkanImageInteropInfo*,
    VkImageViewType,
    VkImageLayout,
]
_LIB.ti_import_vulkan_image.restype = TiImage


def ti_import_vulkan_image(
    runtime: TiRuntime,
    interop_info: ctypes.c_void_p,  # const TiVulkanImageInteropInfo*,
    view_type: VkImageViewType,
    layout: VkImageLayout,
) -> TiImage:
    """
    Function `ti_import_vulkan_image` (1.4.0)

    Imports the Vulkan image owned by Taichi to external procedures.

    Return value: TiImage

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiVulkanImageInteropInfo`):
        view_type (`VkImageViewType`):
        layout (`VkImageLayout`):
    """
    out = _LIB.ti_import_vulkan_image(runtime, interop_info, view_type, layout)
    return TiImage(out)


_LIB.ti_export_vulkan_image.argtypes = [
    TiRuntime,
    TiImage,
    ctypes.c_void_p,  # TiVulkanImageInteropInfo*,
]
_LIB.ti_export_vulkan_image.restype = None


def ti_export_vulkan_image(
    runtime: TiRuntime,
    image: TiImage,
    interop_info: ctypes.c_void_p,  # TiVulkanImageInteropInfo*
) -> None:
    """
    Function `ti_export_vulkan_image` (1.4.0)

    Exports a Vulkan image from external procedures to Taichi.

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        image (`TiImage`):
        interop_info (`TiVulkanImageInteropInfo`):
    """
    out = _LIB.ti_export_vulkan_image(runtime, image, interop_info)
