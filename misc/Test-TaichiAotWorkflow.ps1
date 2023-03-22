param (
    # Skip Taichi installation.
    [switch] $SkipInstall,
    # Debug mode: Generate files in current directory and keep the generated artifacts.
    [switch] $Debug
)

function Install-Taichi {
    if (-not $SkipInstall) {
        & pip install -i https://pypi.taichi.graphics/simple/ taichi-nightly
    }

    # Check Taichi (nightly) installation.
    & ti module -h
    if ($LASTEXITCODE -ne 0) {
        throw "Taichi (nightly) installation failed."
    }
}


function Main {
    param (
        [string] $Arch
    )

    # Test for `ti module build`.

    Write-Host "Generating arange.py..."
    '
import taichi as ti

ti.init(arch=ti.' + $Arch + ')

@ti.aot.export
@ti.kernel
def arange(a: ti.types.ndarray(ti.i32, ndim=1)):
    for i in a:
        a[i] = i
    ' > "arange.py"

    Write-Host "Building arange_module.tcm..."
    & ti module build "arange.py" --output "arange_module.tcm"

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path "arange_module.tcm")) {
        throw "Taichi AOT module build failed."
    }


    # Test for `ti module cppgen`.

    Write-Host "Generating arange.h from arange_module.tcm..."
    & ti module cppgen "arange_module.tcm" --output "arange.h" --namespace "test"

    Write-Host "Generating arange.cpp..."
    '
#include <iostream>
#include <cstdint>
#include <taichi/cpp/taichi.hpp>
#include "arange.h"

int main(int argc, char **argv) {
    ti::Runtime runtime = ti::Runtime(TI_ARCH_' + $ARCH.ToUpper() + ');

    ti::NdArray<int32_t> a = runtime.allocate_ndarray<int32_t>({8}, {}, true);

    test::AotModule_arange_module aot_module = test::AotModule_arange_module::load(runtime, "arange_module.tcm");
    test::Kernel_arange k_arange = aot_module.get_kernel_arange();
    k_arange.set_a(a);
    k_arange.launch();

    runtime.wait();

    std::vector<uint32_t> out(8);
    a.read(out);
    for (int i = 0; i < 8; i++) {
        std::cout << i << " ";
        if (out[i] != i) {
            return 1;
        }
    }

    return 0;
}
    ' > "main.cpp"

    Write-Host "Downloading FindTaichi.cmake from Taichi repository..."
    New-Item -ItemType Directory -Path "cmake" -Force
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/taichi-dev/taichi/master/c_api/cmake/FindTaichi.cmake" -OutFile "cmake/FindTaichi.cmake" -Proxy "http://127.0.0.1:1080"

    Write-Host "Generating CMakeLists.txt..."
    '
project(TestAotWorkflow)
cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Taichi REQUIRED)

add_executable(TestAotWorkflow main.cpp)
target_link_libraries(TestAotWorkflow PRIVATE Taichi::Runtime)

add_custom_command(
    TARGET TestAotWorkflow
    PRE_BUILD
    COMMENT "-- Copy redistributed libraries to output directory (if different): ${Taichi_REDIST_LIBRARIES}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Taichi_REDIST_LIBRARIES} "$<TARGET_FILE_DIR:TestAotWorkflow>")
    ' > "CMakeLists.txt"

    Write-Host "Compiling and running TestAotWorkflow..."
    & cmake -B "build" -GNinja -S "."

    if ($LASTEXITCODE -ne 0) {
        throw "CMake generation failed."
    }

    & cmake --build "build"

    $Executable = Get-ChildItem build -Recurse -File | Where-Object { $_.Name.IndexOf("TestAotWorkflow") -ge 0 } | Select-Object -First 1
    Write-Host $Executable

    if ($LASTEXITCODE -ne 0 -or -not $Executable) {
        throw "Compilation failed."
    }

    & $Executable

    if ($LASTEXITCODE -ne 0) {
        throw "Execution failed."
    }
}

try {
    if ($Debug) {
        $WorkingDir = "./$(New-Guid)"
    } else {
        $TempDir = $(Get-PSDrive -Name "Temp").Root
        $WorkingDir = "$TempDir/$(New-Guid)"
    }
    Write-Host "Working directory: $WorkingDir"
    New-Item -ItemType Directory -Path $WorkingDir -Force

    Push-Location $WorkingDir

    if ($IsMacOS) {
        $Arch = "metal"
    } else {
        $Arch = "vulkan"
    }

    Install-Taichi

    Main -Arch $Arch

    Write-Host "Test passed."
} catch {
    Write-Error $_
} finally {
    Pop-Location
    if (!($Debug)) {
        Remove-Item -Path $WorkingDir -Recurse -Force
    }
}
