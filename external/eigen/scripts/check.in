#!/bin/bash
# check : shorthand for make and ctest -R

if [[ $# != 1 || $1 == *help ]]
then
  echo "usage: $0 regexp"
  echo "  Builds and runs tests matching the regexp."
  echo "  The EIGEN_MAKE_ARGS environment variable allows to pass args to 'make'."
  echo "    For example, to launch 5 concurrent builds, use EIGEN_MAKE_ARGS='-j5'"
  echo "  The EIGEN_CTEST_ARGS environment variable allows to pass args to 'ctest'."
  echo "    For example, with CTest 2.8, you can use EIGEN_CTEST_ARGS='-j5'."
  exit 0
fi

if [ -n "${EIGEN_CTEST_ARGS:+x}" ]
then
  ./buildtests.sh "$1" && ctest -R "$1" ${EIGEN_CTEST_ARGS}
else
  ./buildtests.sh "$1" && ctest -R "$1"
fi
exit $?
