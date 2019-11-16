import csv
from statistics import median
from typing import Tuple, List, Dict, DefaultDict, Optional


def unbatch(
        in_path: str,
        batch_size: int,
        column_names: List[str],
        ) -> List[List[Dict[str, str]]]:
    """unbatch one CSV row of k examples into a length k list of examples"""
    with open(in_path) as file:
        reader = csv.DictReader(file)
        raw_CSV = [row for row in reader]
    submissions: List[List[Dict[str, str]]] = []
    for row in raw_CSV:
        submission = []
        for batch_index in range(batch_size):
            example: Dict[str, str] = {}
            for column in column_names:
                batch_name = column.format(batch_index)
                unbatch_name = column.split('_', 1)[1]
                example[unbatch_name] = row[batch_name]
            example['WorkerId'] = row['WorkerId']
            example['SubmitTime'] = row['SubmitTime']
            submission.append(example)
        submissions.append(submission)
    return submissions


def process(
        submissions: List[List[Dict[str, str]]],
        combined_data: Dict,
        num_labels: int,
        max_skipped: int,
        in_column: str,
        out_column: str,
        ) -> Dict[Tuple[str, str], Dict]:
    # (query, neighbor) -> List[worker_label]
    stats: Dict[Tuple[str, str], List[int]] = DefaultDict(list)
    for submission in submissions:
        worker_skipped = 0
        worker_labels: List[int] = []
        worker_id = submission[0]['WorkerId']
        for entry in submission:
            worker_label: Optional[int] = None
            for label_index in range(num_labels):
                if entry[in_column.format(label_index)] == 'true':
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
            pair_phrases = (entry['query_words'], entry['neighbor_words'])
            stats[pair_phrases].append(worker_label)
        # end one submission of 30 examples
    # end all submissions

    for pair, worker_labels in stats.items():
        flatten_labels: List[int] = []
        for label in range(num_labels):
            count = worker_labels.count(label)
            # row[f'label_{label}_count'] = count
            if label != 0:  # the "can't judge" option
                flatten_labels += [label for _ in range(count)]

        if len(flatten_labels) > 0:
            combined_data[pair][out_column] = median(flatten_labels)
        else:
            combined_data[pair][out_column] = 0
    return combined_data


def main() -> None:
    base_dir = '../../data/evaluation/'
    out_path = base_dir + 'combined_result.csv'
    batch_size = 10
    max_skipped = 5
    num_labels = 6

    deno = unbatch(
        in_path=base_dir + 'results_309.csv',
        batch_size=batch_size,
        column_names=[
            'Input.P{}_query_words',
            'Input.P{}_neighbor_words',
            'Answer.P{}_label_0.on',
            'Answer.P{}_label_1.on',
            'Answer.P{}_label_2.on',
            'Answer.P{}_label_3.on',
            'Answer.P{}_label_4.on',
            'Answer.P{}_label_5.on'])

    query_cono = unbatch(
        in_path=base_dir + 'cono_pilot.csv',
        batch_size=batch_size,
        column_names=[
            'Input.P{}_query_words',
            'Input.P{}_neighbor_words',
            'Answer.P{}_query_label_0.on',
            'Answer.P{}_query_label_1.on',
            'Answer.P{}_query_label_2.on',
            'Answer.P{}_query_label_3.on',
            'Answer.P{}_query_label_4.on',
            'Answer.P{}_query_label_5.on'])

    neighbor_cono = unbatch(
        in_path=base_dir + 'cono_pilot.csv',
        batch_size=batch_size,
        column_names=[
            'Input.P{}_query_words',
            'Input.P{}_neighbor_words',
            'Answer.P{}_neighbor_label_0.on',
            'Answer.P{}_neighbor_label_1.on',
            'Answer.P{}_neighbor_label_2.on',
            'Answer.P{}_neighbor_label_3.on',
            'Answer.P{}_neighbor_label_4.on',
            'Answer.P{}_neighbor_label_5.on'])

    combined_data: DefaultDict[Tuple[str, str], Dict] = DefaultDict(dict)
    combined_data = process(
        deno, combined_data, num_labels, max_skipped,
        in_column='label_{}.on',
        out_column='median_deno')

    combined_data = process(
        query_cono, combined_data, num_labels, max_skipped,
        in_column='query_label_{}.on',
        out_column='median_query_cono')

    combined_data = process(
        neighbor_cono, combined_data, num_labels, max_skipped,
        in_column='neighbor_label_{}.on',
        out_column='median_neighbor_cono')

    column_names = [
        'query_words', 'neighbor_words',
        'median_deno', 'median_query_cono', 'median_neighbor_cono'
    ]
    with open(out_path, 'w') as out_file:
        writer = csv.DictWriter(out_file, column_names)
        writer.writeheader()
        for (query, nieghbor), labels in combined_data.items():
            row = {}
            row['query_words'] = '_'.join(query.split())
            row['neighbor_words'] = '_'.join(nieghbor.split())
            # labels: Dict[str, int]
            row.update(labels)
            writer.writerow(row)


if __name__ == '__main__':
    main()
