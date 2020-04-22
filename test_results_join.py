import os
import argparse
import csv
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir",
                        help="log directory")
    args = parser.parse_args()

    log_dir = args.logdir

    all_subdirs = [log_dir + '/' + d for d in sorted(os.listdir(log_dir)) if os.path.isdir(log_dir + '/' + d)]

    results_list = []

    for sub_dir in all_subdirs:

        print("\nJoining test outcomes in dir:", sub_dir)

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

        df = pd.read_csv(results_file, index_col=None, header=0)

        max_score_idx = df['Score'].idxmax(axis='columns')

        df = df.iloc[range(max_score_idx, max_score_idx + 5)]

        df['TotalTrainTime'] = total_train_time

        # Add empty line
        df = df.append(pd.Series(), ignore_index=True)

        # Append to dfs set
        results_list.append(df)

    df = pd.concat(results_list, axis=0, ignore_index=True, sort=False)

    df.to_csv(os.path.join(log_dir, "all_results.csv"), index=False)


if __name__ == "__main__":
    main()
