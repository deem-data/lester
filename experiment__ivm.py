from lester.ivm.feature_deletion import delete_features
import numpy as np
import torch

def ivm_update(run_id, num_customers):
    pipeline_name = 'lester-gen'
    customers_source_path = f'data/synthetic_customers_{num_customers}.csv'
    mails_source_path = f'data/synthetic_mails_{num_customers}.csv'
    source_column_name = 'mail_subject'
    row_provenance_ids = [2, 4, 6, 8, 9]

    updated_train_data, updated_test_data, updated_X_train, updated_X_test, updated_model = \
        delete_features(pipeline_name, run_id, mails_source_path, source_column_name, customers_source_path, row_provenance_ids)

    updated_train_data.to_csv('.scratchspace/train.csv')
    updated_test_data.to_csv('.scratchspace/test.csv')
    np.save('.scratchspace/X_train.npy', updated_X_train)
    np.save('.scratchspace/X_test.npy', updated_X_test)
    torch.save(updated_model, ".scratchspace/__model.pt")


if __name__ == '__main__':
    import argparse
    import time

    argparser = argparse.ArgumentParser(description='IVM experiments')
    argparser.add_argument('--run_id', required=True)
    argparser.add_argument('--num_customers', required=True)
    argparser.add_argument('--num_repetitions', required=True)
    args = argparser.parse_args()

    for repetition in range(0, int(args.num_repetitions)):
        print(f"# Starting repetition {repetition+1}/{args.num_repetitions} with {args.num_customers} customers")
        start = time.time()
        ivm_update(args.run_id, args.num_customers)
        runtime_in_ms = int((time.time() - start) * 1000)
        print(f"{args.num_customers},{runtime_in_ms}")
