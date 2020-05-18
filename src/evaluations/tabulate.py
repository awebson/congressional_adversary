import csv
import pickle
# import warnings
from pathlib import Path
from typing import List, Dict
import torch
from torch.nn.utils import rnn

from decomposer import Decomposer, DecomposerConfig, LabeledSentences
from recomposer import Recomposer, RecomposerConfig
from evaluations.helpers import lazy_load_recomposers, polarized_words, PE
# from evaluations.euphemism import cherry_pairs

# warnings.simplefilter('ignore')
DEVICE = 'cuda:0'
out_path = '../../analysis/congress.tsv'
# out_path = '../../analysis/cono_space_3bin_all.tsv'



polarized_ids = torch.tensor([w.word_id for w in polarized_words], device=DEVICE)
# crisis_ids = torch.tensor([w.word_id for w in crisis_words], device=DEVICE)

# from evaluations.helpers import all_words  # random sample!
# all_ids = torch.tensor([w.word_id for w in all_words], device=DEVICE)

test_path = Path('../../data/ellie/partisan_sample.cr.txt')
with open(test_path) as file:
    test_words = [word.strip() for word in file]
test_ids = torch.tensor([PE.word_to_id[w] for w in test_words], device=DEVICE)


# # Load Accuracy Dev Sentences
# corpus_path = '../../data/processed/bill_mentions/topic_deno/train_data.pickle'
# with open(corpus_path, 'rb') as corpus_file:
#     preprocessed = pickle.load(corpus_file)
#     dev_seq = rnn.pad_sequence(
#         [torch.tensor(seq) for seq in preprocessed['dev_sent_word_ids']],
#         batch_first=True).to(DEVICE)
#     dev_deno_labels = torch.tensor(preprocessed['dev_deno_labels'], device=DEVICE)
#     dev_cono_labels = torch.tensor(preprocessed['dev_cono_labels'], device=DEVICE)
# del preprocessed


# tabulate_cos_sim(luntz, deno_embed, cono_embed)
# def tabulate_cos_sim(
#         row: Dict,
#         pairs: List[Tuple[str, str]],
#         d_embed: np.ndarray,
#         c_embed: np.ndarray
#         ) -> None:
#     print('w1', 'w2', 'pretrained', 'deno', 'cono', 'diff', sep='\t')
#     for q1, q2 in pairs:
#         pretrained_cs = round(cos_sim(q1, q2, PE_embed), 4)
#         deno_cs = round(cos_sim(q1, q2, d_embed), 4)
#         cono_cs = round(cos_sim(q1, q2, c_embed), 4)
#         diff = round(deno_cs - cono_cs, 4)
#         print(q1, q2, pretrained_cs, deno_cs, cono_cs, diff, sep='\t')


def evaluate(word_ids, suffix, D_model, C_model) -> Dict[str, float]:
    row = {}
    DS_Hdeno = D_model.NN_cluster_homogeneity(word_ids, probe='deno', top_k=10)
    DS_Hcono = D_model.NN_cluster_homogeneity(word_ids, probe='cono', top_k=10)
    CS_Hdeno = C_model.NN_cluster_homogeneity(word_ids, probe='deno', top_k=10)
    CS_Hcono = C_model.NN_cluster_homogeneity(word_ids, probe='cono', top_k=10)

    row['DS Hdeno'] = DS_Hdeno
    row['DS Hcono'] = DS_Hcono
    row['IntraDS Hd - Hc'] = DS_Hdeno - DS_Hcono

    row['CS Hdeno'] = CS_Hdeno
    row['CS Hcono'] = CS_Hcono
    row['IntraCS Hc - Hd'] = CS_Hcono - CS_Hdeno

    row['Inter DS Hd - CS Hd'] = DS_Hdeno - CS_Hdeno
    row['Inter CS Hc - DS Hc'] = CS_Hcono - DS_Hcono

    row['main diagnoal trace'] = (DS_Hdeno + CS_Hcono) / 2  # max all preservation
    row['nondiagnoal entries negative sum'] = (-DS_Hcono - CS_Hdeno) / 2  # min all discarded
    row['flattened weighted sum'] = row['main diagnoal trace'] + row['nondiagnoal entries negative sum']

    row['mean IntraS quality'] = (row['IntraDS Hd - Hc'] + row['IntraCS Hc - Hd']) / 2
    row['mean InterS quality'] = (row['Inter DS Hd - CS Hd'] + row['Inter CS Hc - DS Hc']) / 2
    return {key + f' ({suffix})': val for key, val in row.items()}


generator = lazy_load_recomposers(
    in_dirs=[Path('../../results/congress bill topic/recomposer'),],
    patterns=['*/epoch10.pt', '*/epoch25.pt', '*/epoch50.pt', '*/epoch75.pt', '*/epoch100.pt'],
    device=DEVICE)

debug = 0
table: List[Dict] = []
for D_model, C_model in generator:
    # deno_accuracy, cono_accuracy = model.accuracy(dev_seq, dev_deno_labels, dev_cono_labels)

    if 'pretrained' in D_model.name:
        row = {'model_name': D_model.name}
    else:
        row = {
            'model_name': D_model.name,
            'epoch': D_model.stem,
            'DS delta': D_model.delta,
            'DS gamma': D_model.gamma,
            'DS delta/gamma': D_model.delta / D_model.gamma,
            'CS delta': C_model.delta,
            'CS gamma': C_model.gamma,
            'CS delta/gamma': C_model.delta / C_model.gamma,
            'rho': D_model.config.recomposer_rho,
            'dropout': D_model.config.dropout_p}

    row.update(evaluate(polarized_ids, 'dev', D_model, C_model))
    row.update(evaluate(test_ids, 'test', D_model, C_model))
    # for key, val in row.items():
    #     if isinstance(val, float):
    #         row[key] = round(val, 4)
    table.append(row)

    # if debug > 5:
    #     break
    # debug += 1

with open(out_path, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=table[-1].keys(), dialect=csv.excel_tab)
    writer.writeheader()
    writer.writerows(table)
