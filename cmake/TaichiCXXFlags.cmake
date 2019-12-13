message("Using C++ compiler: " ${CMAKE_CXX_COMPILER})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_ISE_NONE")


if (MINGW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_hypot=hypot")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMS_WIN64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
endif ()

if (MSVC)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/lib)
    set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} /Zc:__cplusplus /std:c++17 /MP /Z7 /D \"_CRT_SECURE_NO_WARNINGS\" /D \"_ENABLE_EXTENDED_ALIGNED_STORAGE\"")
else()
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        message("Clang compiler detected. Using std=c++17.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fsized-deallocation")
    else()
        message(FATAL_ERROR "clang-6/7 is the only supported compiler for Taichi compiler development")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=nehalem -Wall ")
endif ()

if (USE_STDCPP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_PASS_EXCEPTION_TO_PYTHON")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_INCLUDED")

if ($ENV{TC_USE_DOUBLE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_USE_DOUBLE")
    message("Using float64 (double) precision as real")
else()
    message("Using float32 (single) precision as real")
endif()

if (TC_USE_MPI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_USE_MPI")
    message("Using MPI")
endif ()

if (NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()
