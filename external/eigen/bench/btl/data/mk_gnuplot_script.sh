#! /bin/bash
WHAT=$1
DIR=$2
echo $WHAT script generation
cat $WHAT.hh > $WHAT.gnuplot

DATA_FILE=`find $DIR -name "*.dat" | grep $WHAT`

echo plot \\ >> $WHAT.gnuplot

for FILE in $DATA_FILE
do
    LAST=$FILE
done

echo LAST=$LAST

for FILE in $DATA_FILE
do
     if [ $FILE != $LAST ]
     then
	BASE=${FILE##*/} ; BASE=${FILE##*/} ; AVANT=bench_${WHAT}_ ; REDUC=${BASE##*$AVANT} ; TITLE=${REDUC%.dat}
	echo "'"$FILE"'" title "'"$TITLE"'" ",\\" >>  $WHAT.gnuplot
     fi
done
BASE=${LAST##*/} ; BASE=${FILE##*/} ; AVANT=bench_${WHAT}_ ; REDUC=${BASE##*$AVANT} ; TITLE=${REDUC%.dat}
echo "'"$LAST"'" title "'"$TITLE"'" >>  $WHAT.gnuplot

#echo set term postscript color >> $WHAT.gnuplot
#echo set output "'"$WHAT.ps"'" >> $WHAT.gnuplot
echo set term pbm small color >> $WHAT.gnuplot
echo set output "'"$WHAT.ppm"'" >> $WHAT.gnuplot
echo plot \\ >> $WHAT.gnuplot

for FILE in $DATA_FILE
do
     if [ $FILE != $LAST ]
     then
	BASE=${FILE##*/} ; BASE=${FILE##*/} ; AVANT=bench_${WHAT}_ ; REDUC=${BASE##*$AVANT} ; TITLE=${REDUC%.dat}
	echo "'"$FILE"'" title "'"$TITLE"'" ",\\" >>  $WHAT.gnuplot
     fi
done
BASE=${LAST##*/} ; BASE=${FILE##*/} ; AVANT=bench_${WHAT}_ ; REDUC=${BASE##*$AVANT} ; TITLE=${REDUC%.dat}
echo "'"$LAST"'" title "'"$TITLE"'" >>  $WHAT.gnuplot

echo set term jpeg large >> $WHAT.gnuplot
echo set output "'"$WHAT.jpg"'" >> $WHAT.gnuplot
echo plot \\ >> $WHAT.gnuplot

for FILE in $DATA_FILE
do
     if [ $FILE != $LAST ]
     then
	BASE=${FILE##*/} ; BASE=${FILE##*/} ; AVANT=bench_${WHAT}_ ; REDUC=${BASE##*$AVANT} ; TITLE=${REDUC%.dat}
	echo "'"$FILE"'" title "'"$TITLE"'" ",\\" >>  $WHAT.gnuplot
     fi
done
BASE=${LAST##*/} ; BASE=${FILE##*/} ; AVANT=bench_${WHAT}_ ; REDUC=${BASE##*$AVANT} ; TITLE=${REDUC%.dat}
echo "'"$LAST"'" title "'"$TITLE"'" >>  $WHAT.gnuplot


gnuplot -persist < $WHAT.gnuplot

rm $WHAT.gnuplot




