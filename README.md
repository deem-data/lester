Supplemental material for our CIDR submission: _Messy Code Makes Managing ML Pipelines Difficult? Just Let LLMs Rewrite the Code!_

## Abstraction for ML pipelines

We provide the source code for our prototypical implementation of our proposed pipeline abstraction. Below are pointers to the core components of Lester:

 * [Dataframes with row and column provenance tracking](https://github.com/deem-data/lester/blob/main/lester/__init__.py#L33)
 * [Matrix column provenance tracking for estimator/transformers](https://github.com/deem-data/lester/blob/main/lester/feature_provenance.py)
 * [Execution of pipelines for supervised learning](https://github.com/deem-data/lester/blob/main/lester/classification.py)
 * [IVM updates for a small number of changes input values](https://github.com/deem-data/lester/blob/main/lester/ivm/feature_deletion.py)
 * [IVM updates to remove input samples](https://github.com/deem-data/lester/blob/main/lester/ivm/instance_deletion.py) (not covered in submisson)

## Code Rewriting with LLMs
 * We provide the [messy original code for the ML pipeline](https://github.com/deem-data/lester/blob/main/messy_original_pipeline.py) from our running example
 * We detail the [hand-crafted prompts](https://github.com/deem-data/lester/blob/main/llm-based-rewrites.md) that we used for rewriting our example pipeline 
 * We provide the [generated pipeline code](https://github.com/deem-data/lester/blob/main/generated_pipeline_code.py) and mark code locations that needed manual fixing

As detailed in our submission, we consider it future work to streamline this rewriting process with a conversational interface.

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

### User Study

We conduct a small user study to showcase that even basic tasks like computing certain metadata in ML pipelines are difficult for data scientists without system support. We provide the [tasks, code, reference solution, questionaire and participant code](https://github.com/deem-data/lester/blob/main/study.md).

