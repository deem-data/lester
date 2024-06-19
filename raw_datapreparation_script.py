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
