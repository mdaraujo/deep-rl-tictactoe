import os
import argparse
import csv
import json
import pandas as pd

from utils.hyperparams import N_REPEATS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="log directory")
    parser.add_argument('-n', '--n_repeats', type=int, default=N_REPEATS)
    args = parser.parse_args()

    log_dir = args.logdir
    n_repeats = args.n_repeats

    all_subdirs = []

    for root, _, _ in os.walk(log_dir):
        all_subdirs.append(root)

    all_subdirs.sort()

    results_list = []

    count = 0

    for sub_dir in all_subdirs:

        if not sub_dir[-1].isdigit():
            print("\n", sub_dir)
            count = 0
            results_list.append(pd.DataFrame({'Episodes': ['', sub_dir, '', '']}))
            continue

        train_logs_file = os.path.join(sub_dir, 'train_logs.json')

        if not os.path.isfile(train_logs_file):
            print("Train logs file not found: {}".format(train_logs_file))
            continue

        with open(train_logs_file, 'r') as f:
            train_logs = json.load(f)
            total_train_time = train_logs['elapsed_time_h']

        results_file = os.path.join(sub_dir, "test_outcomes.csv")

        if not os.path.isfile(results_file):
            print("Results file not found: {}".format(results_file))
            continue

        print("Joining test outcomes in dir:", sub_dir)

        df = pd.read_csv(results_file, index_col=None, header=0)

        max_score_idx = df['Score'].idxmax(axis='columns')

        df = df.iloc[range(max_score_idx, max_score_idx + 5)]

        df['TotalTrainTime'] = total_train_time

        count += 1

        # Add empty line
        df = df.append(pd.Series(), ignore_index=True)

        if count >= n_repeats:

            # Calculate average
            avg_score = 0
            for i in range(1, n_repeats):
                avg_score += results_list[-i]['Score'][0]
            avg_score += df['Score'][0]
            avg_score /= n_repeats

            df = df.append({'Score': avg_score}, ignore_index=True)

            # Add another empty line
            df = df.append(pd.Series(), ignore_index=True)
            count = 0

        results_list.append(df)

    df = pd.concat(results_list, axis=0, ignore_index=True, sort=False)

    df.to_csv(os.path.join(log_dir, "all_results.csv"), index=False)


if __name__ == "__main__":
    main()
