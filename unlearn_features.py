
pipeline_name = 'lester-gen'
run_id = '05b10a0d-70eb-44e2-a1ec-274bfe022afb'
customers_source_path = 'data/synthetic_customers_1000.csv'
mails_source_path = 'data/synthetic_mails_1000.csv'
source_column_name = 'mail_subject'
row_provenance_ids = [4, 6]

from lester.ivm.feature_deletion import delete_features

updated_train_data, updated_test_data, updated_X_train, updated_X_test, updated_model = \
    delete_features(pipeline_name, run_id, mails_source_path, source_column_name, customers_source_path, row_provenance_ids)

print(updated_train_data.head())
