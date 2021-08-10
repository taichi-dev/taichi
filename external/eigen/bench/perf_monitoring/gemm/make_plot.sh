#!/bin/bash

# base name of the bench
# it reads $1.out
# and generates $1.pdf
WHAT=$1
bench=$2

header="rev "
while read line
do
  if [ ! -z '$line' ]; then
    header="$header  \"$line\""
  fi
done < $bench"_settings.txt"

echo $header > $WHAT.out.header
cat $WHAT.out >> $WHAT.out.header


echo "set title '$WHAT'" > $WHAT.gnuplot
echo "set key autotitle columnhead outside " >> $WHAT.gnuplot
echo "set xtics rotate 1" >> $WHAT.gnuplot

echo "set term pdf color rounded enhanced fontscale 0.35 size 7in,5in" >> $WHAT.gnuplot
echo set output "'"$WHAT.pdf"'" >> $WHAT.gnuplot

col=`cat $bench"_settings.txt" | wc -l`
echo "plot for [col=2:$col+1] '$WHAT.out.header' using 0:col:xticlabels(1) with lines" >> $WHAT.gnuplot
echo " " >>  $WHAT.gnuplot

gnuplot -persist < $WHAT.gnuplot

# generate a png file
# convert -background white -density 120 -rotate 90 -resize 800 +dither -colors 256 -quality 0 $WHAT.ps -background white -flatten  .$WHAT.png

# clean
rm $WHAT.out.header $WHAT.gnuplot