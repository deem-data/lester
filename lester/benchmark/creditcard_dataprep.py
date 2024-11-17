from lester.benchmark import DataprepCodeTransformationTask


class CreditcardDataprepTask(DataprepCodeTransformationTask):

    @property
    def original_code(self):
        return """
import os
from dateutil import parser
from transformers import pipeline
os.environ["TOKENIZERS_PARALLELISM"] = "False"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
        
target_countries = ['UK', 'DE', 'FR']
customer_data = {}

sentiment_predictor = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

def sanitize(text):
    return text.lower()

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
        mail_year = int(raw_date.split("-")[0])
        if mail_year >= 2022:
            if email in customer_data:
                bank, country, is_premium = customer_data[email]
                title = sanitize(mail_subject)
                text = sanitize(mail_text)
                sentiment = sentiment_predictor(mail_text)[0]['label'].lower()
                output_file.write(f"{title}\t{text}\t{bank}\t{country}\t{sentiment}\t{is_premium}\n")        
"""

    def input_arg_names(self):
        return ['customers_file', 'mails_file']

    def input_schemas(self):
        return [['customer_id', 'customer_email', 'bank', 'country', 'level'],
                ['mail_id', 'email', 'raw_date', 'mail_subject', 'mail_text']]

    def output_columns(self):
        return ['title', 'text', 'bank', 'country', 'sentiment', 'is_premium']

    def run_manually_rewritten_code(self, params):

        import lester as lt
        from transformers import pipeline
        from dateutil import parser

        def sanitize(text):
            return text.lower()

        sentiment_predictor = \
            pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

        customer_df = lt.read_csv(params['customers_file'], header=None,
                                  names=['customer_id', 'customer_email', 'bank', 'country', 'level'])
        customer_df = customer_df.filter("country in ['UK', 'DE', 'FR']")
        customer_df = customer_df.project(target_column='is_premium', source_columns=['level'],
                                          func=lambda level: level == 'premium')
        customer_df = customer_df[['customer_email', 'bank', 'country', 'is_premium']]

        mails_df = lt.read_csv(params['mails_file'], header=None,
                               names=['mail_id', 'email', 'raw_date', 'mail_subject', 'mail_text'])
        mails_df = mails_df.project(target_column='mail_date', source_columns=['raw_date'],
                                    func=lambda raw_date: parser.parse(raw_date))
        mails_df = mails_df.filter('mail_date.dt.year >= 2022')
        mails_df = mails_df.filter("mail_text.str.contains('complaint') or mail_text.str.contains('bank account')")

        merged_df = lt.join(mails_df, customer_df, left_on='email', right_on='customer_email')

        # Process and assign new columns
        merged_df = merged_df.project(target_column='title', source_columns=['mail_subject'],
                                      func=sanitize)
        merged_df = merged_df.project(target_column='text', source_columns=['mail_text'],
                                      func=sanitize)
        merged_df = merged_df.project(target_column='sentiment', source_columns=['mail_text'],
                                      func=lambda mail_text: sentiment_predictor(mail_text)[0]['label'].lower())

        result_df = merged_df[['title', 'text', 'bank', 'country', 'sentiment', 'is_premium']]

        return result_df

    def evaluate_transformed_code(self, transformed_code):

        params = {
            'customers_file': 'data/synthetic_customers_100.csv',
            'mails_file': 'data/synthetic_mails_100.csv'
        }

        variables_for_exec = {}
        exec(transformed_code, variables_for_exec)

        generated_result = variables_for_exec['__dataprep'](customers_file=params['customers_file'],
                                                            mails_file=params['mails_file'])

        manual_result = self.run_manually_rewritten_code(params)

        data_diff = manual_result.df[self.output_columns()].compare(generated_result.df[self.output_columns()])
        assert len(data_diff) == 0
