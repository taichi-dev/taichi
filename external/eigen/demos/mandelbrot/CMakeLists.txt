find_package(Qt4 REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (CMAKE_COMPILER_IS_GNUCXX)
   set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
   add_definitions ( "-DNDEBUG" )
endif (CMAKE_COMPILER_IS_GNUCXX)

include_directories( ${QT_INCLUDE_DIR} )

set(mandelbrot_SRCS
    mandelbrot.cpp
)

qt4_automoc(${mandelbrot_SRCS})

add_executable(mandelbrot ${mandelbrot_SRCS})
add_dependencies(demos mandelbrot)

target_link_libraries(mandelbrot ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY})
