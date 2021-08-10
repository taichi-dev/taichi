set noclip points
set clip one
set noclip two
set bar 1.000000
set border 31 lt -1 lw 1.000
set xdata
set ydata
set zdata
set x2data
set y2data
set boxwidth
set dummy x,y
set format x "%g"
set format y "%g"
set format x2 "%g"
set format y2 "%g"
set format z "%g"
set angles radians
set nogrid
set key title ""
set key left top Right noreverse box linetype -2 linewidth 1.000 samplen 4 spacing 1 width 0
set nolabel
set noarrow
# set nolinestyle # deprecated
set nologscale
set logscale x 10
set offsets 0, 0, 0, 0
set pointsize 1
set encoding default
set nopolar
set noparametric
set view 60, 30, 1, 1
set samples 100, 100
set isosamples 10, 10
set surface
set nocontour
set clabel '%8.3g'
set mapping cartesian
set nohidden3d
set cntrparam order 4
set cntrparam linear
set cntrparam levels auto 5
set cntrparam points 5
set size ratio 0 1,1
set origin 0,0
# set data style lines
# set function style lines
set xzeroaxis lt -2 lw 1.000
set x2zeroaxis lt -2 lw 1.000
set yzeroaxis lt -2 lw 1.000
set y2zeroaxis lt -2 lw 1.000
set tics in
set ticslevel 0.5
set tics scale 1, 0.5
set mxtics default
set mytics default
set mx2tics default
set my2tics default
set xtics border mirror norotate autofreq
set ytics border mirror norotate autofreq
set ztics border nomirror norotate autofreq
set nox2tics
set noy2tics
set timestamp "" bottom norotate offset 0,0
set rrange [ * : * ] noreverse nowriteback  # (currently [-0:10] )
set trange [ * : * ] noreverse nowriteback  # (currently [-5:5] )
set urange [ * : * ] noreverse nowriteback  # (currently [-5:5] )
set vrange [ * : * ] noreverse nowriteback  # (currently [-5:5] )
set xlabel "matrix size" offset 0,0
set x2label "" offset 0,0
set timefmt "%d/%m/%y\n%H:%M"
set xrange [ 10 : 1000 ] noreverse nowriteback
set x2range [ * : * ] noreverse nowriteback  # (currently [-10:10] )
set ylabel "MFLOPS" offset 0,0
set y2label "" offset 0,0
set yrange [ * : * ] noreverse nowriteback  # (currently [-10:10] )
set y2range [ * : * ] noreverse nowriteback  # (currently [-10:10] )
set zlabel "" offset 0,0
set zrange [ * : * ] noreverse nowriteback  # (currently [-10:10] )
set zero 1e-08
set lmargin -1
set bmargin -1
set rmargin -1
set tmargin -1
set locale "C"
set xrange [4:1024]

