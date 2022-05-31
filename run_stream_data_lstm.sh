for exp_data in covType poker rialto sea elec mixeddrift hyperplane chess weather
    do
    for method in lstm
    do
    python stream_sgd_wfa.py --exp_data $exp_data --method $method
    done
done

