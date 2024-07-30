from messy_original_pipeline import messy_pipeline

if __name__ == '__main__':
    import argparse
    import time

    argparser = argparse.ArgumentParser(description='Retraining experiments')
    argparser.add_argument('--num_customers', required=True)
    argparser.add_argument('--num_repetitions', required=True)
    args = argparser.parse_args()

    customers_path = f"data/synthetic_customers_{args.num_customers}.csv"
    mails_path = f"data/synthetic_mails_{args.num_customers}.csv"

    for repetition in range(0, int(args.num_repetitions)):
        print(f"# Starting repetition {repetition+1}/{args.num_repetitions} with {args.num_customers} customers")
        start = time.time()
        messy_pipeline(customers_path, mails_path)
        runtime_in_ms = int((time.time() - start) * 1000)
        print(f"{args.num_customers},{runtime_in_ms}")
