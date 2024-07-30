
# Relational data preparation

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

<code>The following code is written in python with for loops and manual data parsing. Please rewrite the code to use a dataframe library called lester. lester has an API similar to pandas and supports the following operations from pandas: 'merge', 'query', 'assign', 'explode', 'rename'. The 'assign' method in lester has two additional parameters: `target_column` and `source_columns`; `target_column` refers to the new column which should be created, while `source_columns` refers to the list of input columns that are used by the expression in `assign`. Please create a single, separate `assign` statement for each new column that is computed. Only respond with python code. Do not iterate over dataframes. The code should contain a single function called `_lester_dataprep`, which returns a single dataframe called `result_df` as result. This final dataframe should have the following columns: title, text, bank, country, sentiment, is_premium</code>

### Prompt 2:  Making sure that local variables and functions are correctly referenced in expressions

<code>In the following Python code, please make sure that all local variables referenced in the query function start with '@':</code>

### Prompt 3:  Using lester's method names instead of Pandas method names

<code>In the following Python code, please rename the function `assign` to `project`, the function `merge` to `join` and the function `query` to `filter`.</code>

### Prompt 4:  Remove hardcoded input paths

<code>The function `_lester_dataprep` in the following Python code reads CSV files from hardcoded paths. Please rewrite the code such that the function `_lester_dataprep` accepts the paths as function arguments. Respond with Python code only.</code>

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

