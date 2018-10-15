message("Using C++ compiler: " ${CMAKE_CXX_COMPILER})

if (TC_DISABLE_SIMD)
    message("SIMD explicitly disabled. This may lead to performance issues.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_ISE_NONE")
else()
    include(${TAICHI_CMAKE_DIR}/OptimizeForArchitecture.cmake)
    OptimizeForArchitecture()
    message("**************************************************")
    message("* CPU feature detection done.")
    if ("${TARGET_ARCHITECTURE}" MATCHES "sandy-bridge")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_ISE_NONE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSANDY_BRIDGE")
        message("* Using Instruction Set Externsion: [None]")
    elseif (USE_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_ISE_AVX2")
        message("* Using Instruction Set Externsion: [AVX2]")
    elseif (USE_AVX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_ISE_AVX")
        message("* Using Instruction Set Externsion: [AVX]")
    elseif (USE_SSE4_2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_ISE_SSE")
        message("* Using Instruction Set Externsion: [SSE]")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_ISE_NONE")
        message("* Using Instruction Set Externsion: [None]")
    endif ()
    message("**************************************************")
endif()


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
            "${CMAKE_CXX_FLAGS} /MP /Z7 /D \"_CRT_SECURE_NO_WARNINGS\" /D \"_ENABLE_EXTENDED_ALIGNED_STORAGE\" /arch:AVX2 -DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED /std:c++14")
else ()
    set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -std=c++14 -march=native\
 -DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED -Wall")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_PASS_EXCEPTION_TO_PYTHON")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_INCLUDED")

if ($ENV{TC_USE_DOUBLE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_USE_DOUBLE")
    message("Using float64 (double) precision as real")
else()
    message("Using float32 (single) precision as real")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
message("Using -D_GLIBCXX_USE_CXX11_ABI=0")

if (TC_USE_MPI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_USE_MPI")
    message("Using MPI")
endif ()

if (NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

