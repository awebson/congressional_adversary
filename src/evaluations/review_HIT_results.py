import csv
import pprint
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
        # 'Input.P{}_accepted_answers',
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


def review_results(
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

    for pair, labels in stats.items():
        # print(pair, labels, sep='\t')
        query, neighbor = pair
        worker_labels = labels
        c0 = worker_labels.count(0)
        c1 = worker_labels.count(1)
        c2 = worker_labels.count(2)
        c3 = worker_labels.count(3)
        c4 = worker_labels.count(4)
        c5 = worker_labels.count(5)
        print(c0, c1, c2, c3, c4, c5, query, neighbor, sep='\t')


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
        # end one submission of 30 examples
        worker_id = submission[0]['WorkerId']
        worker_skipped[worker_id] = num_skipped
    # end all submissions
    pprint.pprint(worker_skipped)
    return worker_skipped


def export_qualified_workers(
        path: str,
        worker_scores: Dict[str, int],
        min_score: int
        ) -> None:
    print('Exporting qualified workers:')
    with open(path, 'w') as file:
        file.write(
            'Worker ID,'
            'UPDATE-Political Euphemism Detection,'
            'UPDATE-Already Taken Political Euphemism Detection\n')  # disqualified
        for worker_id, score in worker_scores.items():
            if score >= min_score:
                print(worker_id, score, sep='\t')
                file.write(f'{worker_id},{score},\n')
            else:
                file.write(f'{worker_id},,{score}\n')


def export_HIT_approvals(
        path: str,
        raw_CSV: List[Dict[str, str]],
        worker_scores: Dict[str, int],
        min_score: int
        ) -> None:
    qualified_msg = (
        "Congrats! You passed our qualification test :) "
        "Your score was {}, while the minimal score is {}. "
        "We'd love if you continue to participate in our "
        "political phrase HITs. Thank you!")
    disqualified_msg = (
        "Your HIT and reward have been approved. "
        "Your score was {}. Unfortunately the minimal qualifying score is {}. "
        "Sorry! This is a hard task even for our own researchers, so no hard feelings. "
        "Thank you for your participation in our research."
    )
    reject_message = "too many skipped"  # score == -1

    with open(path, 'w') as file:
        writer = csv.DictWriter(file, raw_CSV[0].keys())
        writer.writeheader()
        for row in raw_CSV:
            score = worker_scores[row['WorkerId']]
            if score >= min_score:
                row['Approve'] = qualified_msg.format(score, min_score)
                row['RequesterFeedback'] = qualified_msg.format(score, min_score)
            else:
                row['Approve'] = disqualified_msg.format(score, min_score)
                row['RequesterFeedback'] = disqualified_msg.format(score, min_score)
            writer.writerow(row)


def export_high_num_skipped_workers() -> None:
    pass


if __name__ == '__main__':
    base_dir = '../../data/evaluation/'
    in_path = base_dir + 'pilot_results_2.csv'
    batch_size = 10
    # min_score = 20
    max_skipped = 5
    num_labels = 6
    with open(in_path) as file:
        reader = csv.DictReader(file)
        raw_CSV = [row for row in reader]
    data: List[List[Dict[str, str]]] = [
        unbatch(row, batch_size) for row in raw_CSV]

    review_results(data, max_skipped, num_labels, verbose=True)
    # export_HIT_approvals(
    #     base_dir + 'HIT_approvals.csv', raw_CSV, worker_scores, min_score)
    # export_qualified_workers(
    #     base_dir + 'qualified_workers.csv', worker_scores, min_score)

    count_num_skipped(data, num_labels)
