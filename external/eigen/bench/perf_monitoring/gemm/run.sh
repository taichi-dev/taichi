#!/bin/bash

# ./run.sh gemm
# ./run.sh lazy_gemm

# Examples of environment variables to be set:
#   PREFIX="haswell-fma-"
#   CXX_FLAGS="-mfma"

# Options:
#   -up : enforce the recomputation of existing data, and keep best results as a merging strategy
#   -s  : recompute selected changesets only and keep bests

bench=$1

if echo "$*" | grep '\-up' > /dev/null; then
  update=true
else
  update=false
fi

if echo "$*" | grep '\-s' > /dev/null; then
  selected=true
else
  selected=false
fi

global_args="$*"

if [ $selected == true ]; then
 echo "Recompute selected changesets only and keep bests"
elif [ $update == true ]; then
 echo "(Re-)Compute all changesets and keep bests"
else
 echo "Skip previously computed changesets"
fi



if [ ! -d "eigen_src" ]; then
  hg clone https://bitbucket.org/eigen/eigen eigen_src
else
  cd eigen_src
  hg pull -u
  cd ..
fi

if [ ! -z '$CXX' ]; then
  CXX=g++
fi

function make_backup
{
  if [ -f "$1.out" ]; then
    mv "$1.out" "$1.backup"
  fi
}

function merge
{
  count1=`echo $1 |  wc -w`
  count2=`echo $2 |  wc -w`
  
  if [ $count1 == $count2 ]; then
    a=( $1 ); b=( $2 )
    res=""
    for (( i=0 ; i<$count1 ; i++ )); do
      ai=${a[$i]}; bi=${b[$i]}
      tmp=`echo "if ($ai > $bi) $ai else $bi " | bc -l`
      res="$res $tmp"
    done
    echo $res

  else
    echo $1
  fi
}

function test_current 
{
  rev=$1
  scalar=$2
  name=$3
  
  prev=""
  if [ -e "$name.backup" ]; then
    prev=`grep $rev "$name.backup" | cut -c 14-`
  fi
  res=$prev
  count_rev=`echo $prev |  wc -w`
  count_ref=`cat $bench"_settings.txt" |  wc -l`
  if echo "$global_args" | grep "$rev" > /dev/null; then
    rev_found=true
  else
    rev_found=false
  fi
#  echo $update et $selected et $rev_found because $rev et "$global_args"
#  echo $count_rev et $count_ref
  if [ $update == true ] || [ $count_rev != $count_ref ] || ([ $selected == true ] &&  [ $rev_found == true ]); then
    if $CXX -O2 -DNDEBUG -march=native $CXX_FLAGS -I eigen_src $bench.cpp -DSCALAR=$scalar -o $name; then
      curr=`./$name`
      if [ $count_rev == $count_ref ]; then
        echo "merge previous $prev"
        echo "with new       $curr"
      else
        echo "got            $curr"
      fi
      res=`merge "$curr" "$prev"`
#       echo $res
      echo "$rev $res" >> $name.out
    else
      echo "Compilation failed, skip rev $rev"
    fi
  else
    echo "Skip existing results for $rev / $name"
    echo "$rev $res" >> $name.out
  fi
}

make_backup $PREFIX"s"$bench
make_backup $PREFIX"d"$bench
make_backup $PREFIX"c"$bench

cut -f1 -d"#" < changesets.txt | grep -E '[[:alnum:]]' | while read rev
do
  if [ ! -z '$rev' ]; then
    echo "Testing rev $rev"
    cd eigen_src
    hg up -C $rev > /dev/null
    actual_rev=`hg identify | cut -f1 -d' '`
    cd ..
    
    test_current $actual_rev float                  $PREFIX"s"$bench
    test_current $actual_rev double                 $PREFIX"d"$bench
    test_current $actual_rev "std::complex<double>" $PREFIX"c"$bench
  fi
  
done

echo "Float:"
cat $PREFIX"s""$bench.out"
echo " "

echo "Double:"
cat $PREFIX"d""$bench.out"
echo ""

echo "Complex:"
cat $PREFIX"c""$bench.out"
echo ""

./make_plot.sh $PREFIX"s"$bench $bench
./make_plot.sh $PREFIX"d"$bench $bench
./make_plot.sh $PREFIX"c"$bench $bench


