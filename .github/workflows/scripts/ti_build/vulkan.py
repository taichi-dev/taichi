# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform

# -- third party --
# -- own --
from .dep import download_dep
from .misc import banner, get_cache_home, path_prepend
from .python import path_prepend


# -- code --
@banner("Setup Vulkan 1.3.296.0")
def setup_vulkan():
    u = platform.uname()

    # Check if Vulkan SDK is already available
    if u.system == "Windows":
        # Check common Vulkan SDK installation paths
        possible_paths = [
            "C:\\VulkanSDK",
            "C:\\Program Files\\VulkanSDK",
            "C:\\Program Files (x86)\\VulkanSDK",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                # Find the latest version
                versions = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if versions:
                    latest_version = sorted(versions)[-1]
                    vulkan_sdk_path = os.path.join(path, latest_version)
                    os.environ["VULKAN_SDK"] = vulkan_sdk_path
                    os.environ["VK_SDK_PATH"] = vulkan_sdk_path
                    os.environ["VK_LAYER_PATH"] = os.path.join(vulkan_sdk_path, "Bin")
                    path_prepend("PATH", os.path.join(vulkan_sdk_path, "Bin"))
                    print(f"Using existing Vulkan SDK at: {vulkan_sdk_path}")
                    return

    if u.system == "Linux":
        url = "https://sdk.lunarg.com/sdk/download/1.3.296.0/linux/vulkansdk-linux-x86_64-1.3.296.0.tar.xz"
        prefix = get_cache_home() / "vulkan-1.3.296.0"
        download_dep(url, prefix, strip=1)
        sdk = prefix / "x86_64"
        os.environ["VULKAN_SDK"] = str(sdk)
        path_prepend("PATH", sdk / "bin")
        path_prepend("LD_LIBRARY_PATH", sdk / "lib")
        os.environ["VK_LAYER_PATH"] = str(sdk / "share" / "vulkan" / "explicit_layer.d")
    # elif (u.system, u.machine) == ("Darwin", "arm64"):
    # elif (u.system, u.machine) == ("Darwin", "x86_64"):
    elif (u.system, u.machine) == ("Windows", "AMD64"):
        url = "https://sdk.lunarg.com/sdk/download/1.3.296.0/windows/VulkanSDK-1.3.296.0-Installer.exe"
        prefix = get_cache_home() / "vulkan-1.3.296.0"
        download_dep(
            url,
            prefix,
            elevate=True,
            args=[
                "--accept-licenses",
                "--default-answer",
                "--confirm-command",
                "--root",
                prefix,
                "install",
                "com.lunarg.vulkan.sdl2",
                "com.lunarg.vulkan.glm",
                "com.lunarg.vulkan.volk",
                "com.lunarg.vulkan.vma",
                # 'com.lunarg.vulkan.debug',
            ],
        )
        os.environ["VULKAN_SDK"] = str(prefix)
        os.environ["VK_SDK_PATH"] = str(prefix)
        os.environ["VK_LAYER_PATH"] = str(prefix / "Bin")
        path_prepend("PATH", prefix / "Bin")
    else:
        return
