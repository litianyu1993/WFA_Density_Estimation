for n in 100 500 1000
do
  for noi in 0 0.1 1
  do
    for seed in 1 2 3 4 5 6 7 8 9 10
    do
      python HMM_experiments.py --method HMM --hmm_rank 10 --N $n --new_test_data --noise $noi --seed $seed
    done
  done
done