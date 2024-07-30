# Prompts for rewriting the messy imperative example code to our declarative pipeline abstraction

In the following, we detail our handcrated prompts, which we used to transform the messy imperative pipeline code from our example to our declarative abstraction. We manually fed the sequence of prompts into GPT-4 with the respective code outputs of the previous LLM call. Our goal is to showcase the potential of LLMs to turn imperative code into declarative statements, based on a set of hand-crafted prompts, and we do not (yet) present a conversational system to automate this process. We consider it important future work to streamline and generalise the rewriting process to minimise the amount of manual corrections necessary, e.g., via an agent-based conversational approach.

## Relational data preparation

### Messy original code

```python
import os
from dateutil import parser
from transformers import pipeline
os.environ["TOKENIZERS_PARALLELISM"] = "False"


target_countries = ['UK', 'DE', 'FR']
customer_data = {}

sentiment_predictor = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

def matches_usecase(text):
    return "complaint" in text or "bank account" in text 

def sanitize(text):
    return text.lower()

with open("./test.csv", 'w') as output_file:
    with open("data/customers.csv") as file:
        for line in file:
            parts = line.strip().split(',')
            customer_id, customer_email, bank, country, level = parts
            is_premium = (level == 'premium')
            if country in target_countries:
                customer_data[customer_email] = (bank, country, is_premium)
    
    with open("data/mails.csv") as file:
        for line in file:
            parts = line.strip().split(",")
            mail_id, email, raw_date, mail_subject, mail_text = parts
            mail_date = parser.parse(raw_date)
            if mail_date.year >= 2022 and matches_usecase(mail_text):
                if email in customer_data:                
                    bank, country, is_premium = customer_data[email]					
                    title = sanitize(mail_subject)
                    text = sanitize(mail_text)
                    sentiment = sentiment_predictor(mail_text)[0]['label'].lower()
                    output_file.write(f"{title}\t{text}\t{bank}\t{country}\t{sentiment}\t{is_premium}\n")
```


### Prompt 1: Introducing declarative dataframe operations 

```
The following code is written in python with for loops and manual data parsing. Please rewrite the code to use
a dataframe library called lester. lester has an API similar to pandas and supports the following operations
from pandas: 'merge', 'query', 'assign', 'explode', 'rename'. The 'assign' method in lester has two additional
parameters: `target_column` and `source_columns`; `target_column` refers to the new column which should be
created, while `source_columns` refers to the list of input columns that are used by the expression in `assign`.
Please create a single, separate `assign` statement for each new column that is computed. Only respond with python
code. Do not iterate over dataframes. The code should contain a single function called `_lester_dataprep`, which
returns a single dataframe called `result_df` as result. This final dataframe should have the following
columns: [list of output columns insert here]

[code inserted here]
```

### Prompt 2:  Making sure that local variables and functions are correctly referenced in expressions

```
In the following Python code, please make sure that all local variables referenced in the query function
start with '@':

[code inserted here]
```

### Prompt 3:  Using lester's method names instead of Pandas method names

```
In the following Python code, please rename the function `assign` to `project`, the function `merge` to
`join` and the function `query` to `filter`.

[code inserted here]
```

### Prompt 4:  Remove hardcoded input paths

```
The function `_lester_dataprep` in the following Python code reads CSV files from hardcoded paths. Please
rewrite the code such that the function `_lester_dataprep` accepts the paths as function arguments.
Respond with Python code only.

[code inserted here]
```

### Resulting generated code

```python
import lester as lt
from transformers import pipeline
from dateutil import parser

target_countries = ['UK', 'DE', 'FR']
sentiment_predictor = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')


def matches_usecase(text):
    return "complaint" in text or "bank account" in text


def sanitize(text):
    return text.lower()


def _lester_dataprep(customers_path, mails_path):
    # Load customer data
    customer_df = lt.read_csv(customers_path, header=None,
                              names=['customer_id', 'customer_email', 'bank', 'country', 'level'])
    customer_df = customer_df.filter('country in @target_countries')
    customer_df = customer_df.project(target_column='is_premium', source_columns=['level'],
                                      func=lambda row: row['level'] == 'premium')

    # Select relevant columns for merging
    customer_df = customer_df[['customer_email', 'bank', 'country', 'is_premium']]

    # Load mails data
    mails_df = lt.read_csv(mails_path, header=None, names=['mail_id', 'email', 'raw_date', 'mail_subject', 'mail_text'])
    mails_df = mails_df.project(target_column='mail_date', source_columns=['raw_date'],
                                func=lambda row: parser.parse(row['raw_date']))
    mails_df = mails_df.filter('mail_date.dt.year >= 2022')

    mails_df = mails_df.filter('mail_text.apply(@matches_usecase)')

    # Merge dataframes
    merged_df = lt.join(mails_df, customer_df, left_on='email', right_on='customer_email')

    # Process and assign new columns
    merged_df = merged_df.project(target_column='title', source_columns=['mail_subject'],
                                  func=lambda row: sanitize(row['mail_subject']))
    merged_df = merged_df.project(target_column='text', source_columns=['mail_text'],
                                  func=lambda row: sanitize(row['mail_text']))
    merged_df = merged_df.project(target_column='sentiment', source_columns=['mail_text'],
                                  func=lambda row: sentiment_predictor(row['mail_text'])[0]['label'].lower())

    # Select the required columns
    result_df = merged_df[['title', 'text', 'bank', 'country', 'sentiment', 'is_premium']]

    return result_df
```

## Feature encoding

### Messy original code

```python
    import numpy as np
    from sentence_transformers import SentenceTransformer

    sentence_embedder = SentenceTransformer("all-mpnet-base-v2")

    def count_words(text):
        return len(text.split(" "))

    country_indices = {'DE': 0, 'FR': 1, 'UK': 2}

    titles = []
    title_lengths = []
    texts = []
    countries = []


    with open(".scratchspace/__intermediate.csv") as file:
        for line in file:
            parts = line.strip().split("\t")
            title, text, bank, country, sentiment, is_premium = parts

            titles.append(title)
            title_lengths.append(len(title))
            texts.append(text)
            countries.append(country)

    print('### Embedding titles')
    subject_embeddings = sentence_embedder.encode(titles)
    print('### Embedding mails')
    text_embeddings = sentence_embedder.encode(texts)
    title_lengths_column = np.array(title_lengths)
    title_lengths_column = (title_lengths_column - np.mean(title_lengths_column)) / np.std(title_lengths_column)

    country_onehot = np.zeros((len(countries), len(country_indices)))
    for row, country in enumerate(countries):
        country_onehot[row, country_indices[country]] = 1.0


    X = np.concatenate((
        subject_embeddings,
        text_embeddings,
        title_lengths_column.reshape(-1,1),
        country_onehot
    ), axis=1)
```

### Prompt: Introducing declarative sklearn estimator/transformers

<code>The following Python reads a CSV file and manually encodes the data as feature for a machine learning model. Please rewrite the code to use estimator/transformers from scikit-learn and the ColumnTransformer from scikit learn. Only respond with python code. Create a function called `encode_features` which returns an unfitted ColumnTransformer which contains the feature encoding logic. The `encode_features` function should be able to work on data that follows the exact schema of the CSV file.</code>

### Resulting generated code (with manual fixes highlighted)

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sentence_transformers import SentenceTransformer


class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X, y=None):
        X = [elem[0] for elem in X.values]  # NEEDED TO BE MANUALLY ADDED!
        return self.model.encode(X)


class WordCountTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = [elem[0] for elem in X.values]  # NEEDED TO BE MANUALLY ADDED!
        return np.array([len(text.split(" ")) for text in X]).reshape(-1, 1)


def encode_features():
    subject_pipeline = Pipeline([
        ('embedding', SentenceEmbeddingTransformer()),
    ])

    text_pipeline = Pipeline([
        ('embedding', SentenceEmbeddingTransformer()),
    ])

    subject_length_pipeline = Pipeline([
        ('length', WordCountTransformer()),
        ('scaler', StandardScaler()),
    ])

    country_pipeline = Pipeline([
        ('onehot', OneHotEncoder(sparse_output=False))  # NEEDED TO BE MANUALLY EDITED
    ])

    preprocessor = ColumnTransformer([
        ('subject_embedding', subject_pipeline, ['title']),
        ('text_embedding', text_pipeline, ['text']),
        ('subject_length', subject_length_pipeline, ['title']),
        ('country', country_pipeline, ['country'])
    ])

    return preprocessor
```
