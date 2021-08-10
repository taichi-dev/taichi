#!/bin/bash
WHAT=$1
DIR=$2

cat ../gnuplot_common_settings.hh > ${WHAT}.gnuplot

echo "set title " `grep ${WHAT} ../action_settings.txt | head -n 1 | cut -d ";" -f 2` >> $WHAT.gnuplot
echo "set xlabel " `grep ${WHAT} ../action_settings.txt | head -n 1 | cut -d ";" -f 3` " offset 0,0" >> $WHAT.gnuplot
echo "set xrange [" `grep ${WHAT} ../action_settings.txt | head -n 1 | cut -d ";" -f 4` "]" >> $WHAT.gnuplot

if [ $# > 3 ]; then
  if [ "$3" == "tiny" ]; then
    echo "set xrange [2:16]" >> $WHAT.gnuplot
    echo "set nologscale" >> $WHAT.gnuplot
  fi
fi



DATA_FILE=`cat ../order_lib`
echo set term postscript color rounded enhanced >> $WHAT.gnuplot
echo set output "'"../${DIR}/$WHAT.ps"'" >> $WHAT.gnuplot

# echo set term svg color rounded enhanced >> $WHAT.gnuplot
# echo "set terminal svg enhanced size 1000 1000 fname \"Times\" fsize 36" >> $WHAT.gnuplot
# echo set output "'"../${DIR}/$WHAT.svg"'" >> $WHAT.gnuplot

echo plot \\ >> $WHAT.gnuplot

for FILE in $DATA_FILE
do
    LAST=$FILE
done

for FILE in $DATA_FILE
do
    BASE=${FILE##*/} ; BASE=${FILE##*/} ; AVANT=bench_${WHAT}_ ; REDUC=${BASE##*$AVANT} ; TITLE=${REDUC%.dat}

    echo "'"$FILE"'" `grep $TITLE ../perlib_plot_settings.txt | head -n 1 | cut -d ";" -f 2` "\\" >>  $WHAT.gnuplot
    if [ $FILE != $LAST ]
    then
      echo ", \\" >>  $WHAT.gnuplot
    fi
done
echo " " >>  $WHAT.gnuplot

gnuplot -persist < $WHAT.gnuplot

rm $WHAT.gnuplot

ps2pdf ../${DIR}/$WHAT.ps ../${DIR}/$WHAT.pdf
convert -background white -density 120 -rotate 90 -resize 800 +dither -colors 256 -quality 0 ../${DIR}/$WHAT.ps -background white -flatten  ../${DIR}/$WHAT.png

# pstoedit -rotate -90 -xscale 0.8 -yscale 0.8 -centered -yshift -50 -xshift -100  -f plot-svg aat.ps  aat2.svg
