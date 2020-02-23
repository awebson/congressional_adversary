import csv
from typing import List, Dict
import torch

from decomposer import Decomposer, DecomposerConfig
from recomposer import Recomposer, RecomposerConfig
from evaluations.helpers import lazy_load_recomposers
from evaluations.clustering import J_ids

DEVICE = 'cpu'
EVAL_DENO_SPACE = True
out_path = '../../analysis/deno_space.csv'
# out_path = '../../analysis/cono_space.csv'
J_ids = torch.tensor(J_ids, device=DEVICE)

deno_spaces = lazy_load_recomposers(
    in_dirs=['../../results/recomposer', '../../results/sans recomposer'],
    suffixes=['epoch2.pt', 'epoch10.pt', 'epoch50.pt', 'epoch100.pt'],
    get_deno_decomposer=EVAL_DENO_SPACE,
    device=DEVICE)

table: List[Dict] = []
for model in deno_spaces:
    Dh = model.NN_cluster_homogeneity(J_ids, eval_deno=True, top_k=5)
    Ch = model.NN_cluster_homogeneity(J_ids, eval_deno=False, top_k=5)

    # deno_accuracy, cono_accuracy = self.model.accuracy(
    #     self.data.dev_seq.to(self.device),
    #     self.data.dev_deno_labels.to(self.device),
    #     self.data.dev_cono_labels.to(self.device))

    if EVAL_DENO_SPACE:
        if 'pretrained' in model.name:
            row = {
                'model_name': model.name,
                'preserve deno': Dh,
                'remove cono': Ch,
                'diff': Dh - Ch
            }
        else:
            row = {
                'model_name': model.name,
                'suffix': model.suffix,
                r'\delta\sub{D}': model.delta,
                r'\gamma\sub{D}': model.gamma,
                r'\rho': model.config.recomposer_rho,
                'dropout': model.config.dropout_p,
                'preserve deno': Dh,
                'remove cono': Ch,
                'diff': Dh - Ch
            }
    else:  # eval cono space
        if 'pretrained' in model.name:
            row = {
                'model_name': model.name,
                'remove deno': Dh,
                'preserve cono': Ch,
                'diff': Ch - Dh
            }
        else:
            row = {
                'model_name': model.name,
                'suffix': model.suffix,
                r'\delta\sub{C}': model.delta,
                r'\gamma\sub{C}': model.gamma,
                r'\rho': model.config.recomposer_rho,
                'dropout': model.config.dropout_p,
                'remove deno': Dh,
                'preserve cono': Ch,
                'diff': Ch - Dh
            }

    for key, val in row.items():
        if isinstance(val, float):
            row[key] = round(val, 4)
    table.append(row)

with open(out_path, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=table[-1].keys(), dialect=csv.excel_tab)
    writer.writeheader()
    writer.writerows(table)
