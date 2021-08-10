# - Try to find how to link to the standard math library, if anything at all is needed to do.
# On most platforms this is automatic, but for example it's not automatic on QNX.
#
# Once done this will define
#
#  STANDARD_MATH_LIBRARY_FOUND - we found how to successfully link to the standard math library
#  STANDARD_MATH_LIBRARY - the name of the standard library that one has to link to.
#                            -- this will be left empty if it's automatic (most platforms).
#                            -- this will be set to "m" on platforms where one must explicitly
#                               pass the "-lm" linker flag.
#
# Copyright (c) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
# Redistribution and use is allowed according to the terms of the 2-clause BSD license.


include(CheckCXXSourceCompiles)

# a little test program for c++ math functions.
# notice the std:: is required on some platforms such as QNX

set(find_standard_math_library_test_program
"
#include<cmath>
int main(int argc, char **){
  return int(std::sin(double(argc)) + std::log(double(argc)));
}")

# first try compiling/linking the test program without any linker flags

set(CMAKE_REQUIRED_FLAGS "")
set(CMAKE_REQUIRED_LIBRARIES "")
CHECK_CXX_SOURCE_COMPILES(
  "${find_standard_math_library_test_program}"
  standard_math_library_linked_to_automatically
)

if(standard_math_library_linked_to_automatically)

  # the test program linked successfully without any linker flag.
  set(STANDARD_MATH_LIBRARY "")
  set(STANDARD_MATH_LIBRARY_FOUND TRUE)

else()

  # the test program did not link successfully without any linker flag.
  # This is a very uncommon case that so far we only saw on QNX. The next try is the
  # standard name 'm' for the standard math library.

  set(CMAKE_REQUIRED_LIBRARIES "m")
  CHECK_CXX_SOURCE_COMPILES(
    "${find_standard_math_library_test_program}"
    standard_math_library_linked_to_as_m)

  if(standard_math_library_linked_to_as_m)

    # the test program linked successfully when linking to the 'm' library
    set(STANDARD_MATH_LIBRARY "m")
    set(STANDARD_MATH_LIBRARY_FOUND TRUE)

  else()

    # the test program still doesn't link successfully
    set(STANDARD_MATH_LIBRARY_FOUND FALSE)

  endif()

endif()
