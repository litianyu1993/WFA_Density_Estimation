export TABLE=wfa_table.dat

N_cases=$(cat "$TABLE" | wc -l)

for ((i=1; i<=$N_cases; i++))
 do # Using environment variable I_FOR to communicate the case number to individual jobs:
  export I_FOR=$i 
  sbatch run_stream_data.sh
 done

export TABLE=gru_table.dat

N_cases=$(cat "$TABLE" | wc -l)

for ((i=1; i<=$N_cases; i++))
 do # Using environment variable I_FOR to communicate the case number to individual jobs:
  export I_FOR=$i
  sbatch run_stream_data.sh
 done

 export TABLE=lstm_table.dat

N_cases=$(cat "$TABLE" | wc -l)

for ((i=1; i<=$N_cases; i++))
 do # Using environment variable I_FOR to communicate the case number to individual jobs:
  export I_FOR=$i
  sbatch run_stream_data.sh
 done

  export TABLE=no_rec_table.dat

N_cases=$(cat "$TABLE" | wc -l)

for ((i=1; i<=$N_cases; i++))
 do # Using environment variable I_FOR to communicate the case number to individual jobs:
  export I_FOR=$i
  sbatch run_stream_data.sh
 done