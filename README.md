Supplemental material for our CIDR submission: _Messy Code Makes Managing ML Pipelines Difficult? Just Let LLMs Rewrite the Code!_

## Abstraction for ML pipelines

We provide the source code for our prototypical implementation of our proposed pipeline abstraction. Below are pointers to the core components of Lester:

 * [Dataframes with row and column provenance tracking](https://github.com/deem-data/lester/blob/main/lester/__init__.py#L33)
 * [Matrix column provenance tracking for estimator/transformers](https://github.com/deem-data/lester/blob/main/lester/feature_provenance.py)
 * [Execution of pipelines for supervised learning](https://github.com/deem-data/lester/blob/main/lester/classification.py)
 * [IVM updates for a small number of changes input values](https://github.com/deem-data/lester/blob/main/lester/ivm/feature_deletion.py)
 * [IVM updates to remove input samples](https://github.com/deem-data/lester/blob/main/lester/ivm/instance_deletion.py) (not covered in submisson)


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

We conduct a small user study to showcase that even basic tasks like computing certain metadata in ML pipelines are difficult for data scientists without system support.

#### Tasks

Participants are asked to extend the ML pipeline available in this colab notebook: https://colab.research.google.com/drive/1Kk4BTmRcgNffNSJrE3Hv6hIMvm03jRaZ

In particular, they should implement the following tasks:

* **T1** – Assess the group fairness of the pipeline for third-party reviews and non-third-party reviews. The chosen fairness metric is “equal opportunity”, e.g., the difference in false negative rates of the predictions for both groups on the test data.
* **T2** – Track the record usage for the products and ratings relations by computing two boolean arrays for them, which denote which records in the relations have been used to train the model

Note that we provide a reference solution here as part of our materials: https://colab.research.google.com/drive/1UgkzlRJf1gUcN-mvRt9sik0jc0DYHMb5 

#### Questions

* How long did you need for the first task?
* How long did you need for the second task?
* How did you experience the task? Was it clear what you had to do?
* How difficult did you find the tasks? Which tasks were most difficult? What was the biggest challenge?
* Did you have to work on such problems in the past?
* Did you complete the tasks? If not, why not? How long would it take you to finish them?
* Did the pipelines that you worked with in the past have rather higher or rather lower code quality?
* Would you use a tool which can automate these tasks?

#### Partipicant notebooks

We provide the anonymised notebooks of our nine participants [for download](https://github.com/deem-data/lester/raw/main/user-study/participants-notebooks.zip).
