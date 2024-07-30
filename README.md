# Lester


## Experiments

### Benefits of incremental view maintenance for a deployed ML pipeline

 * Generate synthetic data for experimentation with the following jupyter notebook: https://github.com/deem-data/lester/blob/main/utils/generate_synthetic_data.ipynb
   
 * Baseline: Run the original pipeline from scratch as a baseline via
`python experiment__retraining_time.py --num_customers <num_customers> --num_repetitions <num_repetitions>`
 * IVM with lester ( and the run id for the directory with the captured artifacts)
   * Initial execution of pipeline (adjust source paths to point to generated data) 
     `python creditcard_example__initial_execution.py`
   * IVM update of the captured artifacts of the pipeline
`python experiment__ivm.py --run_id <run_id> --num_customers <num_customers> --num_repetitions <num_repetitions>`
