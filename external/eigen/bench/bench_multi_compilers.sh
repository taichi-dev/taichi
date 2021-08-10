#!/bin/bash

if (($# < 2)); then
    echo "Usage: $0 compilerlist.txt benchfile.cpp"
else

compilerlist=$1
benchfile=$2

g=0
source $compilerlist

# for each compiler, compile benchfile and run the benchmark
for (( i=0 ; i<g ; ++i )) ; do
  # check the compiler exists
  compiler=`echo ${CLIST[$i]} | cut -d " " -f 1`
  if [ -e `which $compiler` ]; then
    echo "${CLIST[$i]}"
#     echo "${CLIST[$i]} $benchfile -I.. -o bench~"
#     if [ -e ./.bench ] ; then rm .bench; fi
    ${CLIST[$i]} $benchfile -I.. -o .bench && ./.bench 2> /dev/null
    echo ""
  else
    echo "compiler not found: $compiler"
  fi
done

fi
