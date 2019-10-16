import csv
from typing import List, Dict, Optional


def batch(
        in_path: str,
        out_path: str,
        batch_size: int,
        max_num_batches: Optional[int] = None
        ) -> None:
    with open(in_path) as file:
        reader = csv.DictReader(file)
        data: List[Dict] = [row for row in reader]
    if max_num_batches is None:
        max_num_batches = len(data)

    coulmn_names = [
        f'P{i}_' + key
        for i in range(batch_size)
        for key in data[0].keys()]

    with open(out_path, 'w') as out_file:
        writer = csv.DictWriter(out_file, coulmn_names)
        writer.writeheader()

        data = iter(data)
        num_batches = 0
        try:
            while num_batches <= max_num_batches:
                batch = {}
                for i in range(batch_size):
                    example = next(data)
                    batch.update(
                        {f'P{i}_' + key: val
                         for key, val in example.items()})
                writer.writerow(batch)
                num_batches += 1
        except StopIteration:
            print(f'Exported {num_batches} batches of size {batch_size}')

def main() -> None:
    base_dir = '../../data/evaluation/'

    batch(
        base_dir + 'MTurk.csv',
        base_dir + 'batched_MTurk.csv',
        batch_size=5,
        max_num_batches=490)  # 500 for sandbox, 250k for real HIT


if __name__ == '__main__':
    main()
