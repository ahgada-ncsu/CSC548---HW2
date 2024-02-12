set terminal png size 1600,900 font "arial, 18"
set output sprintf('%s.png', filename)
set grid
set xlabel 'X'
set ylabel 'Y'
plot sprintf('%s.dat', filename) u 1:2 w lp t 'Fn(x)', '' u 1:3 w lp t 'Integral(Fn(x))'