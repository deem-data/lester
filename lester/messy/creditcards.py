import os
from dateutil import parser
from transformers import pipeline
os.environ["TOKENIZERS_PARALLELISM"] = "False"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tempfile
import os


def __dataprep_original(customers_file, mails_file):
    target_countries = ['UK', 'DE', 'FR']
    customer_data = {}

    sentiment_predictor = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

    def sanitize(text):
        return text.lower()

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as output_file:
        with open(customers_file) as file:
            for line in file:
                parts = line.strip().split(',')
                customer_id, customer_email, bank, country, level = parts
                is_premium = (level == 'premium')
                if country in target_countries:
                    customer_data[customer_email] = (bank, country, is_premium)

        with open(mails_file) as file:
            for line in file:
                parts = line.strip().split(",")
                mail_id, email, raw_date, mail_subject, mail_text = parts
                mail_date = parser.parse(raw_date)
                if mail_date.year >= 2022:
                    if email in customer_data:
                        bank, country, is_premium = customer_data[email]
                        title = sanitize(mail_subject)
                        text = sanitize(mail_text)
                        sentiment = sentiment_predictor(mail_text)[0]['label'].lower()
                        output_file.write(f"{title}\t{text}\t{bank}\t{country}\t{sentiment}\t{is_premium}\n")

    result = pd.read_csv(output_file.name, sep='\t', header=None,
                         names=['title', 'text', 'bank', 'country', 'sentiment', 'is_premium'])

    if os.path.exists(output_file.name):
        os.remove(output_file.name)

    return result


def __dataprep_manual(customers_path, mails_path):

    import lester as lt
    from transformers import pipeline
    from dateutil import parser

    def sanitize(text):
        return text.lower()

    sentiment_predictor = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

    customer_df = lt.read_csv(customers_path, header=None,
                              names=['customer_id', 'customer_email', 'bank', 'country', 'level'])
    customer_df = customer_df.filter("country in ['UK', 'DE', 'FR']")
    customer_df = customer_df.project(target_column='is_premium', source_columns=['level'],
                                      func=lambda row: row['level'] == 'premium')
    customer_df = customer_df[['customer_email', 'bank', 'country', 'is_premium']]

    mails_df = lt.read_csv(mails_path, header=None, names=['mail_id', 'email', 'raw_date', 'mail_subject', 'mail_text'])
    mails_df = mails_df.project(target_column='mail_date', source_columns=['raw_date'],
                                func=lambda row: parser.parse(row['raw_date']))
    mails_df = mails_df.filter('mail_date.dt.year >= 2022')
    mails_df = mails_df.filter("mail_text.str.contains('complaint') or mail_text.str.contains('bank account')")

    merged_df = lt.join(mails_df, customer_df, left_on='email', right_on='customer_email')

    # Process and assign new columns
    merged_df = merged_df.project(target_column='title', source_columns=['mail_subject'],
                                  func=lambda row: sanitize(row['mail_subject']))
    merged_df = merged_df.project(target_column='text', source_columns=['mail_text'],
                                  func=lambda row: sanitize(row['mail_text']))
    merged_df = merged_df.project(target_column='sentiment', source_columns=['mail_text'],
                                  func=lambda row: sentiment_predictor(row['mail_text'])[0]['label'].lower())

    result_df = merged_df[['title', 'text', 'bank', 'country', 'sentiment', 'is_premium']]

    return result_df
