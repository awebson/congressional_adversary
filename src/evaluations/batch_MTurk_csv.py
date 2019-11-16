import csv
import random
from typing import List, Dict, Optional

random.seed(42)
remove_underscores = str.maketrans('_', ' ')
underscore_to_plus = str.maketrans('_', '+')

def batch(
        in_path: str,
        out_path: str,
        batch_size: int,
        shuffle: bool,
        max_num_batches: Optional[int] = None,
        export_multi_files: Optional[bool] = False
        ) -> None:
    with open(in_path) as file:
        reader = csv.DictReader(file)
        data: List[Dict] = [row for row in reader]
    if max_num_batches is None:
        max_num_batches = len(data)
    if shuffle:
        random.shuffle(data)

    data[0]['google_query'] = ''
    data[0]['google_neighbor'] = ''
    data[0]['google_both'] = ''
    coulmn_names = [
        f'P{i}_' + key
        for i in range(batch_size)
        for key in data[0].keys()]

    with open(out_path, 'w') as out_file:
        writer = csv.DictWriter(out_file, coulmn_names)
        writer.writeheader()

        data = iter(data)  # type: ignore
        num_batches = 0
        try:
            while num_batches < max_num_batches:
                batch = {}
                for i in range(batch_size):
                    example = next(data)  # type: ignore

                    example['google_query'] = example['query_words'].translate(underscore_to_plus)
                    example['google_neighbor'] = example['neighbor_words'].translate(underscore_to_plus)
                    example['google_both'] = example['google_query'] + '+' + example['google_neighbor']

                    batch.update(
                        {f'P{i}_' + key: val.translate(remove_underscores)
                         for key, val in example.items()})
                writer.writerow(batch)
                num_batches += 1
        except StopIteration:
            print(f'Exported {num_batches} batches of size {batch_size}')
        # print(f'Exported {max_num_batches} batches of size {batch_size}')

def main() -> None:
    base_dir = '../../data/evaluation/'

    batch(
        in_path=base_dir + 'top2_partisan_neighbors.csv',
        out_path=base_dir + 'unabridged_batch.csv',
        batch_size=10,
        shuffle=False,
        max_num_batches=66)  # 500 for sandbox, 250k for real HIT


if __name__ == '__main__':
    main()
