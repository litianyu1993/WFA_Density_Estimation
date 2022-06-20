for n in 100 500 1000
do
  for noi in 0 0.1 1
  do
    for seed in 1 2 3 4 5 6 7 8 9 10
    do
      python HMM_experiments.py --method LSTM --LSTM_epochs 100 --hmm_rank 10 --hankel_epochs 100 --fine_tune_epochs 100 --N $n --load_test_data --noise $noi --seed $seed
    done
  done
done