message("Using C++ compiler: " ${CMAKE_CXX_COMPILER})

include(${TAICHI_CMAKE_DIR}/OptimizeForArchitecture.cmake)
OptimizeForArchitecture()

message("**************************************************")
message("* CPU feature detection done.")
if (USE_AVX2)
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
            "${CMAKE_CXX_FLAGS} /MP /Z7 /D \"_CRT_SECURE_NO_WARNINGS\" /arch:AVX\
            -DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED")
else ()
    set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -std=c++14 -march=native\
            -DGL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_PASS_EXCEPTION_TO_PYTHON")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_INCLUDED")

if (TC_DISABLE_SSE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_DISABLE_SSE")
    message("SSE Disabled")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_DISABLE_SSE")
endif ()

if (TC_USE_DOUBLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_USE_DOUBLE")
    message("Using float64 precision")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_DISABLE_SSE")
endif ()

if (TC_USE_MPI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_USE_MPI")
    message("Using MPI")
endif ()

if (TC_USE_OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_USE_OPENMP")
    message("Using OpenMP")
    if (APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libiomp5")
    elseif (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif ()
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
