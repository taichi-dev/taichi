#! /bin/bash
WHAT=$1
DIR=$2
MINIC=$3
MAXIC=$4
MINOC=$5
MAXOC=$6
prefix=$8

meanstatsfilename=$2/mean.html

WORK_DIR=tmp
mkdir $WORK_DIR

DATA_FILE=`find $DIR -name "*.dat" | grep _${WHAT}`

if [ -n "$DATA_FILE" ]; then

  echo ""
  echo "$1..."
  for FILE in $DATA_FILE
  do
          ##echo hello world
          ##echo "mk_mean_script1" ${FILE}
    BASE=${FILE##*/} ; BASE=${FILE##*/} ; AVANT=bench_${WHAT}_ ; REDUC=${BASE##*$AVANT} ; TITLE=${REDUC%.dat}

    ##echo "mk_mean_script1" ${TITLE}
    cp $FILE ${WORK_DIR}/${TITLE}

  done

  cd $WORK_DIR
  ../main $1 $3 $4 $5 $6 * >> ../$meanstatsfilename
  ../mk_new_gnuplot.sh $1 $2 $7
  rm -f *.gnuplot
  cd ..

  echo '<br/>' >> $meanstatsfilename

  webpagefilename=$2/index.html
  # echo '<h3>'${WHAT}'</h3>'  >> $webpagefilename
  echo '<hr/><a href="'$prefix$1'.pdf"><img src="'$prefix$1'.png" alt="'${WHAT}'" /></a><br/>'  >> $webpagefilename

fi

rm -R $WORK_DIR






