import csv
# import warnings
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm

from decomposer import Decomposer, DecomposerConfig, LabeledSentences
from evaluations.helpers import PE

# warnings.simplefilter('ignore')

in_dir = Path('../../results/CR_topic/overcorrect')
patterns = ['*/epoch5.pt', '*/epoch10.pt', '*/epoch15.pt', '*/epoch30.pt', '*/epoch50.pt']
out_path = in_dir / 'analysis.tsv'
device = 'cuda:0'
# out_path = '../../analysis/cono_space_3bin_all.tsv'

checkpoints: List[Path] = []
for pattern in patterns:
    checkpoints += list(in_dir.glob(pattern))
if len(checkpoints) == 0:
    raise FileNotFoundError(f'No model with path pattern found at {in_dir}?')


rand_path = Path('../../data/ellie/rand_sample.cr.txt')
with open(rand_path) as file:
    rand_words = [word.strip() for word in file if word.strip() in PE.word_to_id]

dev_path = Path('../../data/ellie/partisan_sample_val.cr.txt')
with open(dev_path) as file:
    dev_words = [word.strip() for word in file]

test_path = Path('../../data/ellie/partisan_sample.cr.txt')
with open(test_path) as file:
    test_words = [word.strip() for word in file]


debug = 0
wid = None
table: List[Dict] = []
for path in tqdm(checkpoints):
    tqdm.write(f'Loading {path}')
    payload = torch.load(path, map_location=device)
    model = payload['model']
    config = payload['config']
    row = {
        'path': path.parent.name + '/' + path.name,  # path.parent.name
        'delta': config.delta,
        'gamma': config.gamma,
        # 'max_adversary_loss': config.max_adversary_loss,
        'batch size': config.batch_size,
        'learning rate': config.learning_rate,
    }

    if wid is not None:
        assert wid == model.word_to_id
    else:
        wid = model.word_to_id
        rand_ids = torch.tensor([wid[word] for word in rand_words], device=device)
        dev_ids = torch.tensor([wid[word] for word in dev_words], device=device)
        test_ids = torch.tensor([wid[word] for word in test_words], device=device)


    row.update(model.tabulate(dev_ids, ' (dev)'))
    row.update(model.tabulate(rand_ids, ' (random)'))
    row.update(model.tabulate(test_ids, ' (test)'))
    table.append(row)

    # if debug > 5:
    #     break
    # debug += 1

with open(out_path, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=table[1].keys(), dialect=csv.excel_tab)
    writer.writeheader()
    writer.writerows(table)
