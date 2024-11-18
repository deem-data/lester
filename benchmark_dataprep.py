from lester.benchmark.creditcard_dataprep import CreditcardDataprepTask
from lester.benchmark.yichun_dataprep import YichunDataprepTask
from lester.benchmark.amazonreviews_dataprep import AmazonreviewsDataprepTask

CREDITCARD_DATAPREP_CODE = '''
def __dataprep(customers_file, mails_file):
    import os
    from dateutil import parser
    from transformers import pipeline
    import lester as ld
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    target_countries = ['UK', 'DE', 'FR']
    sentiment_predictor = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

    def sanitize(text):
        return text.lower()

    # Load customer data
    customers_df = ld.read_csv(customers_file, header=None, names=['customer_id', 'customer_email', 'bank', 'country', 'level'], sep=",", parse_dates=False)

    # Filter target countries and create 'is_premium' column
    customers_df = customers_df.filter("country in @target_countries")
    customers_df = customers_df.project('is_premium', ['level'], lambda level: level == 'premium')

    # Select relevant columns
    customers_df = customers_df[['customer_email', 'bank', 'country', 'is_premium']]

    # Load mail data
    mails_df = ld.read_csv(mails_file, header=None, names=['mail_id', 'email', 'raw_date', 'mail_subject', 'mail_text'], sep=",", parse_dates=False)

    # Filter mails from year 2022 onwards
    mails_df = mails_df.project('mail_year', ['raw_date'], lambda raw_date: int(raw_date.split("-")[0]))
    mails_df = mails_df.filter("mail_year >= 2022")

    # Join customer and mail data
    merged_df = ld.join(mails_df, customers_df, left_on='email', right_on='customer_email')

    # Sanitize mail_subject and mail_text
    merged_df = merged_df.project('title', ['mail_subject'], sanitize)
    merged_df = merged_df.project('text', ['mail_text'], sanitize)

    # Predict sentiment
    merged_df = merged_df.project('sentiment', ['mail_text'], lambda mail_text: sentiment_predictor(mail_text)[0]['label'].lower())

    # Select final columns
    result_df = merged_df[['title', 'text', 'bank', 'country', 'sentiment', 'is_premium']]

    return result_df
'''

YICHUN_DATAPREP_CODE = '''
def __dataprep(products_pathes, reviews_pathes):
    import lester as ld

    def read_dataset(products_path, reviews_path, id):
        products_df = ld.read_csv(products_path.format(id), header=None, names=["product_id", "product_category", "product_name"], sep="\t")
        reviews_df = ld.read_csv(reviews_path.format(id), header=None, names=["product_id", "rating", "review"], sep="\t")
        return products_df, reviews_df

    def union_dataset(products_df, reviews_df):
        merged_df = ld.join(products_df, reviews_df, left_on="product_id", right_on="product_id")
        final_df = merged_df[["product_id", "product_category", "product_name", "rating", "review"]]
        return final_df

    products_df_list = []
    reviews_df_list = []
#    for id in range(3):                                                                # MANUALLY REMOVED
#        products_df, reviews_df = read_dataset(products_pathes, reviews_pathes, id)    # MANUALLY REMOVED
    for (products_path, reviews_path) in zip(products_pathes, reviews_pathes):          # MANUALLY ADDED
        products_df, reviews_df = read_dataset(products_path, reviews_path, None)       # MANUALLY ADDED
        products_df_list.append(products_df)
        reviews_df_list.append(reviews_df)

    products_df = ld.union(products_df_list)
    reviews_df = ld.union(reviews_df_list)
    result_df = union_dataset(products_df, reviews_df)

    return result_df
'''

AMAZONREVIEWS_DATAPREP_CODE = '''
def __dataprep(reviewPath):
    import lester as ld
    import numpy as np
    import nltk
    from nltk.tokenize import word_tokenize
    nltk.download('punkt')

    # Convert line from input file into an id/text/label tuple
    def parse_review_label(label):
        return "fake" if label == "__label1__" else "real"

    def pre_process(text):
        # Example preprocessing function (can be customized)
        return " ".join(word_tokenize(text.lower()))

    # Load data into a dataframe
    raw_df = ld.read_csv(reviewPath, header=0, sep='\t', names=['DOC_ID', 'LABEL', 'RATING', 'VERIFIED_PURCHASE',
                                                                'PRODUCT_CATEGORY', 'PRODUCT_ID', 'PRODUCT_TITLE',
                                                                'REVIEW_TITLE', 'REVIEW_TEXT'])

    # Parse the LABEL to real/fake
    raw_df = raw_df.project('parsed_label', ['LABEL'], lambda x: parse_review_label(x[0]))

    # Preprocess the REVIEW_TEXT
    raw_df = raw_df.project('preprocessed_text', ['REVIEW_TEXT'], lambda x: pre_process(x[0]))

    # Select relevant columns and rename them
    result_df = raw_df[['DOC_ID', 'RATING', 'VERIFIED_PURCHASE', 'PRODUCT_CATEGORY', 'REVIEW_TEXT']]
    result_df = result_df.rename({'DOC_ID': 'id', 'RATING': 'rating', 'VERIFIED_PURCHASE': 'verified_purchase',
                                  'PRODUCT_CATEGORY': 'product_category', 'REVIEW_TEXT': 'text'})

    return result_df
'''

print('CreditcardDataprepTask...')
creditcard_task = CreditcardDataprepTask()
creditcard_task.evaluate_transformed_code(CREDITCARD_DATAPREP_CODE)

print('YichunDataprepTask...')
yichun_task = YichunDataprepTask()
yichun_task.evaluate_transformed_code(YICHUN_DATAPREP_CODE)

print('AmazonreviewsDataprepTask...')
amazonreviews_task = AmazonreviewsDataprepTask()
amazonreviews_task.evaluate_transformed_code(AMAZONREVIEWS_DATAPREP_CODE)