for n in 10 100 1000
do
  for r in 5 10 20 40
  do
    python HMM_experiments.py --method LSTM --LSTM_epochs 100 --hmm_rank $r --hankel_epochs 100 --fine_tune_epochs 100 --N $n --new_test_data
    python HMM_experiments.py --method WFA --LSTM_epochs 100 --hmm_rank $r --hankel_epochs 100 --fine_tune_epochs 100 --N $n --load_test_data
  done
done
