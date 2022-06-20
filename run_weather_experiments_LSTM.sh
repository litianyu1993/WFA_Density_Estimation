for n in 200 1000 10000 20000
do
    for seed in 1 2 3 4 5 6 7 8 9 10
    do
      for rank in 10 20 30
      do
        for l in 5
        do
          for method in LSTM
          do
          python real_data.py --exp_data weather --method $method --LSTM_epochs 500 --r $rank --hankel_epochs 300 --fine_tune_epochs 300 --N $n --new_test_data --noise 0. --no_nn_transition --seed $seed --regression_epochs 100 --L $l --LSTM_lr 0.1 --fine_tune_lr 0.1 --hankel_lr 0.1 --regression_lr 0.1 --with_batch_norm
          done
      done
    done
  done
done