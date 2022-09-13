set term png
set output "samples_mixing.png"

set xlabel "Reference Implementation [MOS]"
set ylabel "Torch Implementation [MOS]"
set arrow from 1,1 to 5,5 nohead lc rgb 'red' dt 2

plot 'samples_mixing.dat' with points pt 7 ps 0.5 title "Varying degradation (7000 samples)"
