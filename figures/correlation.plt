set term png
set output "correlation.png"

set style fill solid 0.5 border -1
set style boxplot outliers pointtype 7
set style data boxplot
set boxwidth  0.5
set pointsize 0.5
set ylabel "Pearson's r"
unset xtics
set xtics format " "
set key right bottom

plot 'correlation.dat' using (1):1 title "Varying degradation"
