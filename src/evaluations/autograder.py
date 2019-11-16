import csv
import pprint
from typing import Tuple, List, Dict, Counter, Optional

IGNORE = [
    ('womens rights', 'suffrage'),
    ('inequality', 'racism'),
    ('trickledown', 'cut taxes')
]

AD_HOC_RUBRIC = {
    ('corporate profits', 'earnings'): '3/4/5',
    ('washington spending', 'military spending'): '3/2',
    ('waterboarding', 'interrogation'): '3/4'
}

def unbatch(
        batch: Dict[str, str],  # one CSV row
        batch_size: int
        ) -> List[Dict[str, str]]:
    """unbatch one CSV row of k examples into a length k list of examples"""
    column_names = [
        'Input.P{}_query_words',
        'Input.P{}_neighbor_words',
        'Input.P{}_accepted_answers',
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
        submission.append(example)
    return submission


def grade_qualification(
        submissions: List[List[Dict[str, str]]],  # each submission is a list of CSV row
        max_skipped: int,
        num_labels: int,
        verbose: bool = False
        ) -> Dict[str, int]:
    label_column = 'label_{}.on'
    correct_counter: Counter[Tuple[str, str]] = Counter()
    incorrect_counter: Counter[Tuple[str, str]] = Counter()

    # (query, neighbor) -> (accepted_answer, List[worker_label])
    stats: Dict[Tuple[str, str], Tuple[str, List[int]]] = {}
    worker_scores: Dict[str, int] = {}
    for submission in submissions:
        worker_skipped = 0
        worker_labels: List[int] = []

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

        worker_id = submission[0]['WorkerId']
        if worker_skipped > max_skipped:
            worker_scores[worker_id] = -1
            print(f'Exceeded max_skipped: {worker_id}')
            continue  # to the next worker submission

        score = 0
        incorrects = []
        for entry, worker_label in zip(submission, worker_labels):
            if worker_label == 0:  # the "can't judge" option
                continue  # to the next phrase pair
            pair_phrases = (entry['query_words'], entry['neighbor_words'])

            if pair_phrases in IGNORE:
                continue
            if pair_phrases in AD_HOC_RUBRIC:
                entry['accepted_answers'] = AD_HOC_RUBRIC[pair_phrases]

            if pair_phrases not in stats:
                stats[pair_phrases] = (entry['accepted_answers'], [])
            stats[pair_phrases][1].append(worker_label)
            accepted_answers: List[str] = entry['accepted_answers'].split('/')

            # if accepted_answers == ['']:  # HACK
            #     # print('Missing accepted answer:', pair_phrases)
            #     accepted_answers = ['1']

            if str(worker_label) in accepted_answers:
                score += 1
                correct_counter[pair_phrases] += 1
            else:
                incorrects.append((*pair_phrases, worker_label))
                incorrect_counter[pair_phrases] += 1
        # end one submission of 30 examples

        worker_scores[worker_id] = score
        if verbose:
            print(score, worker_skipped, worker_id, incorrects)
        else:
            print(score, end=', ')
    # end all submissions

    print('Correct:')
    pprint.pprint(correct_counter)
    print('Incorrect:')
    pprint.pprint(incorrect_counter)

    for pair, labels in stats.items():
        # print(pair, labels, sep='\t')
        query, neighbor = pair
        accepted_answers, worker_labels = labels
        c0 = worker_labels.count(0)
        c1 = worker_labels.count(1)
        c2 = worker_labels.count(2)
        c3 = worker_labels.count(3)
        c4 = worker_labels.count(4)
        c5 = worker_labels.count(5)
        print(accepted_answers, c1, c2, c3, c4, c5, query, neighbor, sep='\t')

    return worker_scores


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
    in_path = base_dir + 'qual_0.csv'
    batch_size = 30
    min_score = 20
    max_skipped = 5
    num_labels = 6
    with open(in_path) as file:
        reader = csv.DictReader(file)
        raw_CSV = [row for row in reader]
    data: List[List[Dict[str, str]]] = [
        unbatch(row, batch_size) for row in raw_CSV]

    worker_scores = grade_qualification(data, max_skipped, num_labels, verbose=False)
    export_HIT_approvals(
        base_dir + 'HIT_approvals.csv', raw_CSV, worker_scores, min_score)
    export_qualified_workers(
        base_dir + 'qualified_workers.csv', worker_scores, min_score)

    # count_num_skipped(data, num_labels)
