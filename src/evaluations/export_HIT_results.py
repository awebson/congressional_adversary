import csv
import pprint
from statistics import median
from collections import defaultdict
from typing import Tuple, List, Dict, Counter, Optional


def unbatch(
        batch: Dict[str, str],  # one CSV row
        batch_size: int
        ) -> List[Dict[str, str]]:
    """unbatch one CSV row of k examples into a length k list of examples"""
    column_names = [
        'Input.P{}_query_words',
        'Input.P{}_neighbor_words',
        'Answer.P{}_label_0.on',
        'Answer.P{}_label_1.on',
        'Answer.P{}_label_2.on',
        'Answer.P{}_label_3.on',
        'Answer.P{}_label_4.on',
        'Answer.P{}_label_5.on'
    ]
    submission = []
    for batch_index in range(batch_size):
        example: Dict[str, str] = {}
        for column in column_names:
            batch_name = column.format(batch_index)
            unbatch_name = column.split('_', 1)[1]
            example[unbatch_name] = batch[batch_name]
        example['WorkerId'] = batch['WorkerId']
        example['SubmitTime'] = batch['SubmitTime']
        submission.append(example)
    return submission


def export_results(
        out_path: str,
        submissions: List[List[Dict[str, str]]],  # each submission is a list of CSV row
        max_skipped: int,
        num_labels: int,
        verbose: bool = False
        ) -> None:
    label_column = 'label_{}.on'
    # (query, neighbor) -> List[worker_label]
    stats: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for submission in submissions:
        worker_skipped = 0
        worker_labels: List[int] = []
        worker_id = submission[0]['WorkerId']
        if verbose:
            print(submission[0]['SubmitTime'], worker_id)

        for entry in submission:
            worker_label: Optional[int] = None
            for label_index in range(num_labels):
                if entry[label_column.format(label_index)] == 'true':
                    worker_label = label_index
            if worker_label is None:  # the worker didn't select any option
                worker_label = 0
            if worker_label == 0:
                worker_skipped += 1
            worker_labels.append(worker_label)

        if worker_skipped > max_skipped:
            print(f'Exceeded max_skipped: {worker_id}')
            continue  # to the next worker submission

        for entry, worker_label in zip(submission, worker_labels):
            # if worker_label == 0:  # the "can't judge" option
            #     continue  # to the next phrase pair
            pair_phrases = (entry['query_words'], entry['neighbor_words'])
            stats[pair_phrases].append(worker_label)

            if verbose:
                print(worker_label, pair_phrases, sep='\t')
        # end one submission of 30 examples
        if verbose:
            print()
    # end all submissions

    columns = [
        f'label_{label}_count' for label in range(num_labels)
    ] + ['median_label', 'query_words', 'neighbor_words']
    with open(out_path, 'w') as file:
        writer = csv.DictWriter(file, columns)
        writer.writeheader()
        for pair, worker_labels in stats.items():
            row = {}
            row['query_words'], row['neighbor_words'] = pair

            flatten_labels = []
            for label in range(num_labels):
                count = worker_labels.count(label)
                row[f'label_{label}_count'] = count

                if label != 0:  # the "can't judge" option
                    flatten_labels += [label for _ in range(count)]

            row['median_label'] = median(flatten_labels)

            writer.writerow(row)


def count_num_skipped(
        submissions: List[List[Dict[str, str]]],  # each submission is a list of CSV row
        num_labels: int
        ) -> Dict[str, int]:
    label_column = 'label_{}.on'
    worker_skipped: Dict[str, int] = {}
    for submission in submissions:
        num_skipped = 0
        for entry in submission:
            worker_label: Optional[int] = None
            for label_index in range(num_labels):
                if entry[label_column.format(label_index)] == 'true':
                    worker_label = label_index
            if worker_label is None:  # the worker didn't select any option
                worker_label = 0
            if worker_label == 0:
                num_skipped += 1
        # end one submission
        worker_id = submission[0]['WorkerId']
        worker_skipped[worker_id] = num_skipped
    # end all submissions
    print('Worker: Number of Skipped')
    pprint.pprint(worker_skipped)
    return worker_skipped


if __name__ == '__main__':
    base_dir = '../../data/evaluation/'
    in_path = base_dir + 'tuesday_results.csv'
    out_path = base_dir + 'unbatched_results.csv'
    batch_size = 10
    # min_score = 20
    max_skipped = 5
    num_labels = 6
    with open(in_path) as file:
        reader = csv.DictReader(file)
        raw_CSV = [row for row in reader]
    data: List[List[Dict[str, str]]] = [
        unbatch(row, batch_size) for row in raw_CSV]

    export_results(out_path, data, max_skipped, num_labels, verbose=False)
    # export_HIT_approvals(
    #     base_dir + 'HIT_approvals.csv', raw_CSV, worker_scores, min_score)
    # export_qualified_workers(
    #     base_dir + 'qualified_workers.csv', worker_scores, min_score)

    count_num_skipped(data, num_labels)
