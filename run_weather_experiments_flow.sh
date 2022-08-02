for n in 20000
do
  for seed in 1
    do
      for method in flow
        do
          python real_data.py --method $method --N $n  --seed $seed
        done
    done
done