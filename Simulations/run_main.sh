#! /bin/zsh

: 'This is a silly shell script that runs main.py multiple times with different seeds'

# topo="Frontier_BETA_2.0_N=8_miniBokun"
topo="Prune_miniBokun"

echo "zsh version ${ZSH_VERSION}..."
# for loop {start..stop..step}
for seed in 13 21 47 50 91; do
	echo "====================runnning simulation with seed: ${seed}===================="
    name="${topo}_EO_rng=${seed}_auto" # Name of the model, "{Dataset}
    python3.9 main.py ${name} ${seed}
done 
