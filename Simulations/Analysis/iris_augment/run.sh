#! /bin/zsh

: 'This is a silly shell script that runs replot_LPU.py multiple times'

topo="Bokun"

postfixPT="_results.csv"
postfixLPU="_LPU_results.csv"
pkl="10x10_MNIST_2x${topo}_0.00dB_Loss.pkl"
pklname="${pkl: 0:-4}" # without the extension .pkl
PTcsv="${pklname}${postfixPT}"
LPUcsv="${pklname}${postfixLPU}"

echo "zsh version ${ZSH_VERSION}..."
# for loop {start..stop..step}
for i in {1..4..1}; do
	echo "runnning simulation ${i}:"
    prefix="Double${topo}MNIST${i}"
    cd ../../
    python3.9 replot_LPU.py
    cd Analysis/iris_augment/10x10_MNIST
    mv -v $PTcsv $prefix$postfixPT
    mv -v $LPUcsv $prefix$postfixLPU
    mv -v Plots Plots${i}
    cd ..
done 
