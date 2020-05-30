import csv
import pickle
# import warnings
from pathlib import Path
from typing import List, Dict
import torch
from torch.nn.utils import rnn

from decomposer import Decomposer, DecomposerConfig, LabeledSentences
from recomposer import Recomposer, RecomposerConfig
from evaluations.helpers import lazy_load_recomposers, PE
# from evaluations.euphemism import cherry_pairs

# warnings.simplefilter('ignore')
DEVICE = 'cuda:0'
in_dir = Path('../../results/CR_topic/search')
generator = lazy_load_recomposers(
    [in_dir, Path('../../results/CR_topic/search greek')],
    patterns=[
        '*/epoch5.pt', '*/epoch10.pt', '*/epoch25.pt', '*/epoch50.pt',
        '*/epoch75.pt', '*/epoch90.pt', '*/epoch100.pt'],
    # patterns=['*/epoch10.pt', '*/epoch20.pt', '*/epoch50.pt', '*/epoch80.pt', '*/epoch100.pt'],
    # patterns=['*/epoch10.pt', '*/epoch50.pt', '*/epoch100.pt'],
    # patterns=['*/epoch*.pt', ],
    device=DEVICE)
out_path = in_dir / 'summary.tsv'


rand_path = Path('../../data/ellie/rand_sample.cr.txt')
with open(rand_path) as file:
    rand_words = [word.strip() for word in file if word.strip() in PE.word_to_id]
# print(len(rand_words))
dev_path = Path('../../data/ellie/partisan_sample_val.cr.txt')
with open(dev_path) as file:
    dev_words = [word.strip() for word in file]
test_path = Path('../../data/ellie/partisan_sample.cr.txt')
with open(test_path) as file:
    test_words = [word.strip() for word in file]


rand_ids = torch.tensor([PE.word_to_id[w] for w in rand_words], device=DEVICE)
dev_ids = torch.tensor([PE.word_to_id[w] for w in dev_words], device=DEVICE)
test_ids = torch.tensor([PE.word_to_id[w] for w in test_words], device=DEVICE)

debug = 0
table: List[Dict] = []
for model in generator:
    if 'pretrained' in model.name:
        row = {'model_name': model.name}
    else:
        D_model = model.deno_decomposer
        C_model = model.cono_decomposer
        row = {
            'model_name': model.name,
            'epoch': D_model.stem,
            'DS delta': D_model.delta,
            'DS gamma': D_model.gamma,
            # 'DS delta/gamma': D_model.delta / D_model.gamma,
            'CS delta': C_model.delta,
            'CS gamma': C_model.gamma,
            # 'CS delta/gamma': C_model.delta / C_model.gamma,
            'rho': D_model.config.recomposer_rho,
            'dropout': D_model.config.dropout_p}

    row.update(model.tabulate(dev_ids, ' (dev)'))
    row.update(model.tabulate(rand_ids, ' (random)'))
    row.update(model.tabulate(test_ids, ' (test)'))
    table.append(row)
    # if debug > 5:
    #     break
    # debug += 1

columns = table[1].keys()
# columns = list(table[1].keys()) + list(table[0].keys())
with open(out_path, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=columns, dialect=csv.excel_tab)
    writer.writeheader()
    writer.writerows(table)
