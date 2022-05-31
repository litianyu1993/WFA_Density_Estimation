for exp_data in weather
    do
    for method in gru
    do
    python stream_sgd_wfa.py --exp_data $exp_data --method $method
    done
done

