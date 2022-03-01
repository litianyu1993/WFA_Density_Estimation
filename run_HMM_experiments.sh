for n in 10 100 1000 10000
do
  for r in 2 4 6 8 10 15 20 25 30 40 50
  do
    python HMM_experiments.py --method LSTM --LSTM_epochs 200 --hmm_rank $r --hankel_epochs 200 --fine_tune_epochs 200 --N $n
    python HMM_experiments.py --method WFA --LSTM_epochs 200 --hmm_rank $r --hankel_epochs 200 --fine_tune_epochs 200 --N $n
  done
done
