
pipeline_name = 'lester-gen'
run_id = 'a76c9ee6-2e26-48c5-807a-5732ced9e88c'
customers_source_path = 'data/synthetic_customers_10000.csv'
mails_source_path = 'data/synthetic_mails_10000.csv'
source_column_name = 'mail_subject'
row_provenance_ids = [2, 4, 6, 8, 9]

from lester.ivm.feature_deletion import delete_features

updated_train_data, updated_test_data, updated_X_train, updated_X_test, updated_model = \
    delete_features(pipeline_name, run_id, mails_source_path, source_column_name, customers_source_path, row_provenance_ids)

print(updated_train_data.head())
