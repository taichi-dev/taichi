message("Using C++ compiler: " ${CMAKE_CXX_COMPILER})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_ISE_NONE")

option(BUILD_WITH_ADDRESS_SANITIZER "Build with clang address sanitizer" OFF)

include(CheckCXXCompilerFlag) # For `check_cxx_compiler_flag`.

if (BUILD_WITH_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
endif()

if (MINGW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_hypot=hypot")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMS_WIN64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
endif ()

# Do not enable lto for APPLE since it made linking extremely slow.
if (WIN32)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -flto=thin")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -flto=thin")
    elseif (MSVC)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} /Gy")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} /Gy")
        if (TI_WITH_LTO)
            set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} /GL")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} /GL")
            set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
            set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
        endif()
    endif()
endif()

if (WIN32)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/lib)
    if (MSVC)
        # C++17, and C++ conformance
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus /Zc:inline /std:c++17")
        # Linker & object related flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /bigobj")
        # Debugging (generate PDB files)
        if (TI_GENERATE_PDB)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi /Zf")
        endif()
        # Performance and optimizations
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Oi")
        # C4244: conversion from 'type1' to 'type2', possible loss of data
        # C4267: conversion from 'size_t' to 'type', possible loss of data
        # C4624: destructor was implicitly defined as deleted because a base class destructor is inaccessible or deleted
        # These warnings are not emitted on Clang (mostly within LLVM source code)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267 /wd4624 /nologo /D \"_CRT_SECURE_NO_WARNINGS\" /D \"_ENABLE_EXTENDED_ALIGNED_STORAGE\"")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fsized-deallocation -target x86_64-pc-windows-msvc")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -gcodeview")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -gcodeview")
    endif()
else()
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        message("Clang compiler detected. Using std=c++17.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fsized-deallocation -Wno-deprecated-declarations -Wno-shorten-64-to-32")
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        message("GNU compiler detected. Using std=c++17.")
        message(WARNING "It is detected that you are using gcc as the compiler. This is an experimental feature. Consider adding -DCMAKE_CXX_COMPILER=clang argument to CMake to switch to clang (or MSVC on Windows).")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fsized-deallocation -Wno-class-memaccess -Wno-comment -Wno-sign-compare")
    else()
        message("Invalid compiler ${CMAKE_CXX_COMPILER_ID} detected.")
        message(FATAL_ERROR "clang and MSVC are the only supported compilers for Taichi compiler development. Consider using 'cmake -DCMAKE_CXX_COMPILER=clang' if you are on Linux.")
    endif()

    # [Global] CXX compilation option to enable all warnings.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")

    # Due to limited CI coverage, -Werror is only turned on with Clang-compiler for now.
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        if (NOT ANDROID) # (penguinliong) Blocking builds on Android.
            # [Global] CXX compilation option to treat all warnings as errors.
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror ")
        endif()
    endif()

    # [Global] By default, CXX compiler will throw a warning if it decides to ignore an attribute, for example "[[ maybe unused ]]".
    # However, this behaviour diverges across different compilers (GCC/CLANG), as well as different compiler versions.
    # Therefore we disable such warnings for now.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes ")

    # [Global] Clang warns if a C++ pointer's nullability wasn't marked explicitly (__nonnull, nullable, ...).
    # Nullability seems to be a clang-specific feature, thus we disable this warning.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nullability-completeness ")

    # [Global] Disable warning for unused-private-field for convenience in development.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field ")

    # [Global] By evaluating "constexpr", compiler throws a warning for functions known to be dead at compile time.
    # However, some of these "constexpr" are debug flags and will be manually enabled upon debugging.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unneeded-internal-declaration ")

    # FIXME: Check why Android don't support check_cxx_compiler_flag
    if (NOT ANDROID)
        check_cxx_compiler_flag("-Wno-unqualified-std-cast-call" CXX_HAS_Wno_unqualified_std_cast_call)
        if (${CXX_HAS_Wno_unqualified_std_cast_call})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unqualified-std-cast-call ")
        endif()

        check_cxx_compiler_flag("-Wno-unused-but-set-variable" CXX_HAS_Wno_unused_but_set_variable)
        if (${CXX_HAS_Wno_unused_but_set_variable})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable ")
        endif()
    endif()
endif ()

# (penguinliong) When building for iOS with Xcode `CMAKE_SYSTEM_PROCESSOR`
# is empty
if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "")
    if ("arm64" STREQUAL CMAKE_OSX_ARCHITECTURES)
        set(CMAKE_SYSTEM_PROCESSOR "arm64")
    endif()
endif()

message("Building for processor ${CMAKE_SYSTEM_PROCESSOR}")
if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "AMD64" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "amd64")
    if (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D \"TI_ARCH_x64\"")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_ARCH_x64")
        if ("arm64" IN_LIST CMAKE_OSX_ARCHITECTURES)
            # TODO: (penguinliong) Will probably need this in a future version
            # of Clang. Clang11 doesn't recognize this.
            #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
        else()
            message("Setting -march=nehalem for x86_64 processors")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=nehalem")
        endif()
    endif()
    set(ARCH "x64")
elseif ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_ARCH_ARM")
    set(ARCH "arm64")
elseif ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_ARCH_x86")
    set(ARCH "x86")
else()
    message(FATAL_ERROR "Unknown processor type ${CMAKE_SYSTEM_PROCESSOR}")
endif()
set(HOST_ARCH ${ARCH} CACHE INTERNAL "Host arch")

if (USE_STDCPP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_PASS_EXCEPTION_TO_PYTHON")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_INCLUDED")

if ($ENV{TI_USE_DOUBLE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_USE_DOUBLE")
    message("Using float64 (double) precision as real")
else()
    message("Using float32 (single) precision as real")
endif()

if (TI_USE_MPI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_USE_MPI")
    message("Using MPI")
endif ()

if (APPLE)
    # FIXME: (penguinliong) Shift to automatic reference counting in the future?
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-objc-arc")

    if (XCODE)
        set(XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES)
        # FIXME: (penguinliong) Workaround the overwhelming truncation errors
        # compiling to iOS.
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shorten-64-to-32")
    endif()
endif()
