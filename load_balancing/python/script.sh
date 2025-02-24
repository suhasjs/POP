LOGFILE_DIR=/mnt/solvers/outputs/pop_lb

run_config() {
  echo "OMP_NUM_THREADS=8 python load_balancer_runner.py --numShards $1 --numServers $2 --benchmark $3 --numRounds $4 --load $5 --logfile ${LOGFILE_DIR}/${6}_${1}x${2}x${4}_${7}.pkl"
}

run_config 1024 64 "base-lp-relaxed" 100 "stateless" "lprelaxed" "stateless"
run_config 1024 64 "base-lp-relaxed" 100 "stateful --percentChange 10" "lprelaxed" "stateful_10"

run_config 2048 128 "base-lp-relaxed" 100 "stateless" "lprelaxed" "stateless"
run_config 2048 128 "base-lp-relaxed" 100 "stateful --percentChange 10" "lprelaxed" "stateful_10"