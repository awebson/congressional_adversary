import csv
from pathlib import Path
from typing import Set, List, Dict
import torch

from old_congress import Decomposer, Recomposer, RecomposerConfig
from evaluations.helpers_CR import lazy_load_en_masse, PE, WTI

# warnings.simplefilter('ignore')
DEVICE = 'cuda:0'
in_dir = Path('../../results/CR_skip/GM2')
checkpoints = lazy_load_en_masse(
    in_dir,
    patterns=['*/epoch1.pt', '*/epoch2.pt', '*/epoch4.pt', '*/epoch6.pt', '*/epoch8.pt', '*/epoch10.pt', '*/epoch20.pt'],
    # patterns=['*/epoch1.pt', '*/epoch5.pt', '*/epoch10.pt', '*/epoch15.pt', '*/epoch20.pt', '*/epoch25.pt', '*/epoch30.pt'],
    # patterns=['*/epoch*.pt', ],
    device=DEVICE
)
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

rand_ids = torch.tensor([WTI[w] for w in rand_words], device=DEVICE)
dev_ids = torch.tensor([WTI[w] for w in dev_words], device=DEVICE)
test_ids = torch.tensor([WTI[w] for w in test_words], device=DEVICE)

# with open('../../data/processed/bill_mentions/topic_deno/train_data.pickle', 'rb') as file:
with open('../../data/processed/bill_mentions/title_deno_context5/train_data.pickle', 'rb') as file:
    import pickle
    extra_ground = pickle.load(file)['grounding']

# import IPython
# IPython.embed()
# raise SystemExit

debug = 0
table: List[Dict] = []
for model in checkpoints:
    D_model = model.deno_decomposer
    C_model = model.cono_decomposer

    for word in D_model.grounding.keys():
        D_model.grounding[word]['majority_deno'] = extra_ground[word]['majority_deno']
    C_model.grounding = D_model.grounding

    if 'pretrained' in model.name:
        row = {'model_name': model.name}
    else:
        row = {
            'model_name': model.name,
            # 'epoch': D_model.stem,
            'DS delta': D_model.delta,
            'DS gamma': D_model.gamma,
            # 'DS delta/gamma': D_model.delta / D_model.gamma,
            'CS delta': C_model.delta,
            'CS gamma': C_model.gamma,
            # 'CS delta/gamma': C_model.delta / C_model.gamma,
            'rho': model.rho
        }
    # D_model.deno_grounding.update(rand_deno)
    # D_model.deno_grounding.update(dev_deno)
    # D_model.deno_grounding.update(test_deno)
    # C_model.deno_grounding.update(rand_deno)
    # C_model.deno_grounding.update(dev_deno)
    # C_model.deno_grounding.update(test_deno)

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
