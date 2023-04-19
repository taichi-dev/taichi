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

function New-AotModule {
    param(
        [string] $Arch,
        [string] $Name,
        [string] $PythonSource
    )

    Write-Host "Generating $Name.py..."

    $PythonSource > "$Name.py"

    Write-Host "Building $Name.tcm..."
    & ti module build "$Name.py" --output "$Name.tcm"

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$Name.tcm")) {
        throw "Taichi AOT module build failed."
    }
}


function Main {
    param (
        [string] $Arch
    )

    # Test for `ti module build`.

    $PythonSource = '
import taichi as ti

ti.init(arch=ti.' + $Arch + ')

@ti.aot.export
@ti.kernel
def arange(a: ti.types.ndarray(ti.i32, ndim=1)):
    for i in a:
        a[i] = i
'
    New-AotModule -Arch $Arch -Name "arange" -PythonSource $PythonSource

    # Test for `ti module build` with template args.

    $PythonSource = '
import taichi as ti

ti.init(arch=ti.' + $Arch + ')

@ti.aot.export_as("fill_zero_i32", template_types={
    "a": ti.types.ndarray(ti.i32, ndim=1)
})
@ti.kernel
def fill_zero(a: ti.types.ndarray()):
    for i in a:
        a[i] = 0
'
    New-AotModule -Arch $Arch -Name "fill_zero" -PythonSource $PythonSource

    # Test for `ti module cppgen`.

    Write-Host "Generating arange.h from arange.tcm..."
    & ti module cppgen "arange.tcm" --output "arange.h" --namespace "test"
    & ti module cppgen "fill_zero.tcm" --output "fill_zero.h" --namespace "test" --bin2c

    Write-Host "Generating arange.cpp..."
    '
#include <iostream>
#include <cstdint>
#include <taichi/cpp/taichi.hpp>
#include "arange.h"
#include "fill_zero.h"

int main(int argc, char **argv) {
    ti::Runtime runtime = ti::Runtime(TI_ARCH_' + $ARCH.ToUpper() + ');

    ti::NdArray<int32_t> a = runtime.allocate_ndarray<int32_t>({8}, {}, true);

    test::AotModule_arange arrange_module = test::AotModule_arange::load(runtime, "arange.tcm");
    test::Kernel_arange k_arange = arrange_module.get_kernel_arange();
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

    test::AotModule_fill_zero fill_zero_module = test::AotModule_fill_zero::load(runtime, "fill_zero.tcm");
    test::Kernel_fill_zero k_fill_zero = fill_zero_module.get_kernel_fill_zero();
    k_fill_zero.set_a(a);
    k_fill_zero.launch();

    runtime.wait();

    std::vector<uint32_t> out2(8);
    a.read(out2);
    for (int i = 0; i < 8; i++) {
        std::cout << i << " ";
        if (out2[i] != 0) {
            return 2;
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
