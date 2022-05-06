for n in 1000 10000
do
  for seed in 1 2 3
    do
      for method in flow
        do
          python real_data.py --method $method --N $n  --seed $seed
        done
    done
done