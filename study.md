## Exploratory user study

We conduct a small user study to showcase that even basic tasks like computing certain metadata in ML pipelines are difficult for data scientists without system support.

### Tasks

Participants are asked to extend the ML pipeline available in this colab notebook: https://colab.research.google.com/drive/1Kk4BTmRcgNffNSJrE3Hv6hIMvm03jRaZ

In particular, they should implement the following tasks:

* **T1** – Assess the group fairness of the pipeline for third-party reviews and non-third-party reviews. The chosen fairness metric is “equal opportunity”, e.g., the difference in false negative rates of the predictions for both groups on the test data.
* **T2** – Track the record usage for the products and ratings relations by computing two boolean arrays for them, which denote which records in the relations have been used to train the model

Note that we provide a reference solution here as part of our materials: https://colab.research.google.com/drive/1UgkzlRJf1gUcN-mvRt9sik0jc0DYHMb5 

### Questions

* How long did you need for the first task?
* How long did you need for the second task?
* How did you experience the task? Was it clear what you had to do?
* How difficult did you find the tasks? Which tasks were most difficult? What was the biggest challenge?
* Did you have to work on such problems in the past?
* Did you complete the tasks? If not, why not? How long would it take you to finish them?
* Did the pipelines that you worked with in the past have rather higher or rather lower code quality?
* Would you use a tool which can automate these tasks?

### Partipicant notebooks
