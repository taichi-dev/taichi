#!/bin/bash

black='\E[30m'
red='\E[31m'
green='\E[32m'
yellow='\E[33m'
blue='\E[34m'
magenta='\E[35m'
cyan='\E[36m'
white='\E[37m'

if [ -f $2 ]; then
  data=$2
  if [ -f $1.summ ]; then rm $1.summ; fi
  if [ -f $1.snap ]; then rm $1.snap; fi
else
  data=$1
fi

if ! ./$1 < $data > /dev/null 2> .runtest.log ; then
  echo -e  $red Test $1 failed: $black
  echo -e $blue
  cat .runtest.log
  echo -e $black
  exit 1
else
  if [ -f $1.summ ]; then
    if [ `grep "FATAL ERROR" $1.summ | wc -l` -gt 0 ]; then
      echo -e  $red "Test $1 failed (FATAL ERROR, read the file $1.summ for details)" $black
      echo -e $blue
      cat .runtest.log
      echo -e $black
      exit 1;
    fi

    if [ `grep "FAILED THE TESTS OF ERROR-EXITS" $1.summ | wc -l` -gt 0 ]; then
      echo -e  $red "Test $1 failed (FAILED THE TESTS OF ERROR-EXITS, read the file $1.summ for details)" $black
      echo -e $blue
      cat .runtest.log
      echo -e $black
      exit 1;
    fi      
  fi
  echo -e $green Test $1 passed$black
fi
