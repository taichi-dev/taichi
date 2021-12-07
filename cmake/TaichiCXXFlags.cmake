message("Using C++ compiler: " ${CMAKE_CXX_COMPILER})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_ISE_NONE")

option(BUILD_WITH_ADDRESS_SANITIZER "Build with clang address sanitizer" OFF)

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

if (WIN32)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/lib)
    if (MSVC)
        set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} /Zc:__cplusplus /std:c++17 /bigobj /wd4244 /wd4267 /nologo /Zi /D \"_CRT_SECURE_NO_WARNINGS\" /D \"_ENABLE_EXTENDED_ALIGNED_STORAGE\"")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fsized-deallocation -target x86_64-pc-windows-msvc")
    endif()
else()
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        message("Clang compiler detected. Using std=c++17.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fsized-deallocation")
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        message("GNU compiler detected. Using std=c++17.")
        message(WARNING "It is detected that you are using gcc as the compiler. This is an experimental feature. Consider adding -DCMAKE_CXX_COMPILER=clang argument to CMake to switch to clang (or MSVC on Windows).")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fsized-deallocation -Wno-class-memaccess -Wno-comment -Wno-sign-compare")
    else()
        message("Invalid compiler ${CMAKE_CXX_COMPILER_ID} detected.")
        message(FATAL_ERROR "clang and MSVC are the only supported compilers for Taichi compiler development. Consider using 'cmake -DCMAKE_CXX_COMPILER=clang' if you are on Linux.")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")
endif ()

message("Building for processor ${CMAKE_SYSTEM_PROCESSOR}")
if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "AMD64" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "amd64")
    if (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D \"TI_ARCH_x64\"")
    else()
        message("Setting -march=nehalem for x86_64 processors")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=nehalem -DTI_ARCH_x64")
    endif()
    set(ARCH "x64")
elseif ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64" OR "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTI_ARCH_ARM")
    set(ARCH "arm64")
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
