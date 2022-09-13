set term png
set output "max_error.png"

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

plot 'max_error.dat' i 0 @hist ls 1 title "Varying degradation"
