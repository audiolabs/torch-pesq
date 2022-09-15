set term svg size 1024,512
set output "compare_reference.svg"

set multiplot layout 1, 3 ;

set title "Scatter plot";

set xlabel "Reference Implementation [MOS]"
set ylabel "Torch Implementation [MOS]"
set arrow from 1,1 to 5,5 nohead lc rgb 'red' dt 2

plot 'samples_mixing.dat' with points pt 7 ps 0.5 title ""

set title "Boxplot of correlation coefficient";

unset arrow
set style fill solid 0.5 border -1
set style boxplot outliers pointtype 7
set style data boxplot
set boxwidth  0.5
set pointsize 0.5
set ylabel "Pearson's r"
unset xlabel
unset xtics
#set xtics format " "

plot 'correlation.dat' using (1):1 title ""

set title "Distribution of max error";
binwidth = 0.01
binstart = 0.0

# set width of single bins in histogram
set boxwidth 0.9*binwidth
# set fill style of bins
set style fill solid 0.5
# define macro for plotting the histogram
hist = 'u (binwidth*(floor(($1-binstart)/binwidth)+0.5)+binstart):(1.0) smooth freq w boxes'
set xlabel "Max Error [MOS]"
set xrange [0:0.9]
set ylabel "Occurrences"
set key right top
set xtics 

plot 'max_error.dat' i 0 @hist ls 1 title "Varying degradation"
unset multiplot
