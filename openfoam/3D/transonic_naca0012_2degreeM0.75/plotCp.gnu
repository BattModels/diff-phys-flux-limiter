set term png
set output 'Cp2dM0.75.png'

set yl 'Cp'
set xl 'x/c'

set xr [0:1]

plot 'cpset1.csv' u 1:2 tit 'exp' pt 22, 'postProcessing/samplePwall/surface/0.22/p_patch_aerofoil.raw' u 1:(-(($4-100000)/(0.5*1.17*(259.5**2 + 8.97**2))))  tit 'sim' pt 27
