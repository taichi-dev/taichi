#!/bin/bash

if [ $# < 1 ]; then
  echo "Usage: $0 working_directory [tiny|large [prefix]]"
else

mkdir -p $1
##cp ../libs/*/*.dat $1

mode=large
if [ $# > 2 ]; then
  mode=$2
fi
if [ $# > 3 ]; then
  prefix=$3
fi

EIGENDIR=`cat eigen_root_dir.txt`

webpagefilename=$1/index.html
meanstatsfilename=$1/mean.html

echo ''  > $meanstatsfilename
echo ''  > $webpagefilename
echo '<p><strong>Configuration</strong>'  >> $webpagefilename
echo '<ul>'\
  '<li>' `cat /proc/cpuinfo | grep "model name" | head -n 1`\
  '  (' `uname -m` ')</li>'\
  '<li> compiler: ' `cat compiler_version.txt` '</li>'\
  '<li> eigen3: ' `hg identify -i $EIGENDIR` '</li>'\
  '</ul>' \
  '</p>'  >> $webpagefilename

source mk_mean_script.sh axpy $1 11 2500 100000 250000  $mode $prefix
source mk_mean_script.sh axpby $1 11 2500 100000 250000 $mode $prefix
source mk_mean_script.sh matrix_vector $1 11 50 300 1000 $mode $prefix
source mk_mean_script.sh atv $1 11 50 300 1000 $mode $prefix
source mk_mean_script.sh matrix_matrix $1 11 100 300 1000 $mode $prefix
source mk_mean_script.sh aat $1 11 100 300 1000 $mode $prefix
# source mk_mean_script.sh ata $1 11 100 300 1000 $mode $prefix
source mk_mean_script.sh trmm $1 11 100 300 1000 $mode $prefix
source mk_mean_script.sh trisolve_vector $1 11 100 300 1000 $mode $prefix
source mk_mean_script.sh trisolve_matrix $1 11 100 300 1000 $mode $prefix
source mk_mean_script.sh cholesky $1 11 100 300 1000 $mode $prefix
source mk_mean_script.sh partial_lu_decomp $1 11 100 300 1000 $mode $prefix
source mk_mean_script.sh tridiagonalization $1 11 100 300 1000 $mode $prefix
source mk_mean_script.sh hessenberg $1 11 100 300 1000 $mode $prefix
source mk_mean_script.sh symv $1 11 50 300 1000 $mode $prefix
source mk_mean_script.sh syr2 $1 11 50 300 1000 $mode $prefix
source mk_mean_script.sh ger $1 11 50 300 1000 $mode $prefix
source mk_mean_script.sh rot $1 11 2500 100000 250000 $mode $prefix
source mk_mean_script.sh complete_lu_decomp $1 11 100 300 1000 $mode $prefix

fi

## compile the web page ##

#echo `cat footer.html` >> $webpagefilename