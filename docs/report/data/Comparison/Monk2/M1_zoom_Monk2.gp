#!/usr/bin/gnuplot -persist
#
#    
#    	G N U P L O T
#    	Version 5.2 patchlevel 2    last modified 2017-11-01 
#    
#    	Copyright (C) 1986-1993, 1998, 2004, 2007-2017
#    	Thomas Williams, Colin Kelley and many others
#    
#    	gnuplot home:     http://www.gnuplot.info
#    	faq, bugs, etc:   type "help FAQ"
#    	immediate help:   type "help"  (plot window: hit 'h')
# set terminal qt 0 font "Sans,9"
# set output
unset clip points
set clip one
unset clip two
set errorbars front 1.000000 
set border 31 front lt black linewidth 1.000 dashtype solid
set zdata 
set ydata 
set xdata 
set y2data 
set x2data 
set boxwidth
set style fill  empty border
set style rectangle back fc  bgnd fillstyle   solid 1.00 border lt -1
set style circle radius graph 0.02 
set style ellipse size graph 0.05, 0.03 angle 0 units xy
set dummy x, y
set format x "% h" 
set format y "% h" 
set format x2 "% h" 
set format y2 "% h" 
set format z "% h" 
set format cb "% h" 
set format r "% h" 
set ttics format "% h"
set timefmt "%d/%m/%y,%H:%M"
set angles radians
set tics back
set grid nopolar
set grid xtics nomxtics ytics nomytics noztics nomztics nortics nomrtics \
 nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics
set grid layerdefault   lt 0 linecolor 0 linewidth 0.500 dashtype solid,  lt 0 linecolor 0 linewidth 0.500 dashtype solid
unset raxis
set theta counterclockwise right
set style parallel front  lt black linewidth 2.000 dashtype solid
set key title "" center
set key fixed right top vertical Right noreverse enhanced autotitle nobox
set key noinvert samplen 4 spacing 1 width 0 height 0 
set key maxcolumns 0 maxrows 0
set key noopaque
unset label
unset arrow
set style increment default
unset style line
unset style arrow
set style histogram clustered gap 2 title textcolor lt -1
unset object
set style textbox transparent margins  1.0,  1.0 border  lt -1 linewidth  1.0
set offsets 0, 0, 0, 0
set pointsize 1
set pointintervalbox 1
set encoding default
unset polar
unset parametric
unset decimalsign
unset micro
unset minussign
set view 60, 30, 1, 1
set view azimuth 0
set rgbmax 255
set samples 100, 100
set isosamples 10, 10
set surface 
unset contour
set cntrlabel  format '%8.3g' font '' start 5 interval 20
set mapping cartesian
set datafile separator whitespace
unset hidden3d
set cntrparam order 4
set cntrparam linear
set cntrparam levels auto 5
set cntrparam points 5
set size ratio 0 1,1
set origin 0,0
set style data points
set style function lines
unset xzeroaxis
unset yzeroaxis
unset zzeroaxis
unset x2zeroaxis
unset y2zeroaxis
set xyplane relative 0.5
set tics scale  1, 0.5, 1, 1, 1
set mxtics default
set mytics default
set mztics default
set mx2tics default
set my2tics default
set mcbtics default
set mrtics default
set nomttics
set xtics border in scale 1,0.5 mirror norotate  autojustify
set xtics  norangelimit autofreq 
set ytics border in scale 1,0.5 mirror norotate  autojustify
set ytics  norangelimit autofreq 
set ztics border in scale 1,0.5 nomirror norotate  autojustify
set ztics  norangelimit autofreq 
unset x2tics
unset y2tics
set cbtics border in scale 1,0.5 mirror norotate  autojustify
set cbtics  norangelimit autofreq 
set rtics axis in scale 1,0.5 nomirror norotate  autojustify
set rtics  norangelimit autofreq 
unset ttics
set title "M1 zoom comparison Monk 2" 
set title  font ",16" norotate
set timestamp bottom 
set timestamp "" 
set timestamp  font "" norotate
set trange [ * : * ] noreverse nowriteback
set urange [ * : * ] noreverse nowriteback
set vrange [ * : * ] noreverse nowriteback
set xlabel "iteration" 
set xlabel  font ",14" textcolor lt -1 norotate offset -1.5
set x2label "" 
set x2label  font "" textcolor lt -1 norotate
set xrange [ * : * ] noreverse nowriteback
set x2range [ * : * ] noreverse nowriteback
set ylabel "error" 
set ylabel  font ",14" textcolor lt -1 rotate
set y2label "" 
set y2label  font "" textcolor lt -1 rotate
set yrange [ * : * ] noreverse nowriteback
set y2range [ * : * ] noreverse nowriteback
set zlabel "" 
set zlabel  font "" textcolor lt -1 norotate
set zrange [ * : * ] noreverse nowriteback
set cblabel "" 
set cblabel  font "" textcolor lt -1 rotate
set cbrange [ * : * ] noreverse nowriteback
set rlabel "" 
set rlabel  font "" textcolor lt -1 norotate
set rrange [ * : * ] noreverse nowriteback
unset logscale
unset jitter
set zero 1e-08
set lmargin  +9
set bmargin  -1
set rmargin  -1
set tmargin  -1
set locale "it_IT.UTF-8"
set pm3d explicit at s
set pm3d scansautomatic
set pm3d interpolate 1,1 flush begin noftriangles noborder corners2color mean
set pm3d nolighting
set palette positive nops_allcF maxcolors 0 gamma 1.5 color model RGB 
set palette rgbformulae 7, 5, 15
set colorbox default
set colorbox vertical origin screen 0.9, 0.2 size screen 0.05, 0.6 front  noinvert bdefault
set style boxplot candles range  1.50 outliers pt 7 separation 1 labels auto unsorted
set loadpath 
set fontpath 
set psdir
set fit brief errorvariables nocovariancevariables errorscaling prescale nowrap v5
set key left bottom
set key font ",14"
set tics font ", 13"
set logscale y 10
set format y "10^{%L}"

minimumLBFGS = 0.090313721066937039
minimumMGD = 0.013742702235183110
minimumNMGD = 0.022516608901728817
minimumPBM = 0.000006979035522109

f(x) = x
GNUTERM = "qt"
x = 0.0
GPFUN_f = "f(x) = x"

actualResidual(lossValue, optimalMinimum) = (lossValue - optimalMinimum)

# add smooth bezier to the end to have curves more smooth
plot [:200][] "../../LBFGS/Monk2/LBFGS_L1_0.3_Monk2_Results_SMI.txt" using 0:(actualResidual($1,minimumLBFGS)) w lines title "LBGFS 3e-4" lt rgb "red" lw 2, "../../MGD/Monk2/NM/NMGD_0.6_L2_0.0003_Monk2_Results_SMI.txt" using 0:(actualResidual($1,minimumNMGD)) w lines title "NMGD 3e-4" lt rgb "black" lw 2, "../../MGD/Monk2/M/MGD_0.9_L1_0.0003_Monk2_Results_SMI" using 0:(actualResidual($1,minimumMGD)) w lines title "MGD 3e-4" lt rgb "blue" lw 2, "../../PBM/Monk2/PBM_L1_0.3_Monk2_Results_SMI.txt" using 0:(actualResidual($1,minimumPBM)) w lines title "PBM 3e-4" lt rgb "#005A32" lw 2
#  EOF