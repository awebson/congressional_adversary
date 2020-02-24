import csv
import pickle
import warnings
from pathlib import Path
from typing import List, Dict
import torch
from torch.nn.utils import rnn

from decomposer import Decomposer, DecomposerConfig, LabeledSentences
from recomposer import Recomposer, RecomposerConfig
from evaluations.helpers import lazy_load_recomposers, get_partisan_words

warnings.simplefilter('ignore')
DEVICE = 'cuda:0'
EVAL_DENO_SPACE = True
out_path = '../../analysis/deno_space_F0.tsv'
# out_path = '../../analysis/cono_space_F0.tsv'

deno_spaces = lazy_load_recomposers(
    in_dirs=[Path('../../results/recomposer'), Path('../../results/sans recomposer')],
    patterns=['*/epoch2.pt', '*/epoch10.pt', '*/epoch50.pt', '*/epoch100.pt'],
    get_deno_decomposer=EVAL_DENO_SPACE,
    device=DEVICE)

# # Debug Path
# out_path = '../../analysis/debug.tsv'
# deno_spaces = lazy_load_recomposers(
#     in_dirs=Path('../../results/sans recomposer'),
#     patterns='*/epoch50.pt',
#     get_deno_decomposer=EVAL_DENO_SPACE,
#     device=DEVICE)

# Sample Partisan Words
capitalism = get_partisan_words(min_skew=0.75, max_skew=1, min_freq=0)
socialism = get_partisan_words(min_skew=0, max_skew=0.25, min_freq=0)
# neoliberal_shills = get_partisan_words(min_skew=)
chauvinism = get_partisan_words(min_skew=0.75, max_skew=1, min_freq=100)
bernie_bros = get_partisan_words(min_skew=0.75, max_skew=1, min_freq=100)

polarized_words = capitalism + socialism
crisis_words = chauvinism + bernie_bros
polarized_ids = torch.tensor([w.word_id for w in polarized_words], device=DEVICE)
crisis_ids = torch.tensor([w.word_id for w in crisis_words], device=DEVICE)
print(len(polarized_words))
print(len(crisis_words))

# Load Accuracy Dev Sentences
corpus_path = '../../data/processed/bill_mentions/topic_deno/train_data.pickle'
with open(corpus_path, 'rb') as corpus_file:
    preprocessed = pickle.load(corpus_file)
    dev_seq = rnn.pad_sequence(
        [torch.tensor(seq) for seq in preprocessed['dev_sent_word_ids']],
        batch_first=True).to(DEVICE)
    dev_deno_labels = torch.tensor(preprocessed['dev_deno_labels'], device=DEVICE)
    dev_cono_labels = torch.tensor(preprocessed['dev_cono_labels'], device=DEVICE)
del preprocessed

table: List[Dict] = []
for model in deno_spaces:
    HFreq_Hdeno = model.NN_cluster_homogeneity(crisis_ids, eval_deno=True, top_k=5)
    HFreq_Hcono = model.NN_cluster_homogeneity(crisis_ids, eval_deno=False, top_k=5)
    LFreq_Hdeno = model.NN_cluster_homogeneity(polarized_ids, eval_deno=True, top_k=5)
    LFreq_Hcono = model.NN_cluster_homogeneity(polarized_ids, eval_deno=False, top_k=5)
    deno_accuracy, cono_accuracy = model.accuracy(dev_seq, dev_deno_labels, dev_cono_labels)

    if 'pretrained' in model.name:
        row = {'model_name': model.name}
    else:
        row = {
            'model_name': model.name,
            'epoch': model.stem,
            r'\delta\sub{D}': model.delta,
            r'\gamma\sub{D}': model.gamma,
            r'\rho': model.config.recomposer_rho,
            'dropout': model.config.dropout_p}

    if EVAL_DENO_SPACE:
        row['deno dev accuracy'] = deno_accuracy
        row['LFreq preserve deno'] = LFreq_Hdeno
        row['HFreq preserve deno'] = HFreq_Hdeno
        row['LFreq remove cono'] = LFreq_Hcono
        row['HFreq remove cono'] = HFreq_Hcono
        row['LFreq diff'] = LFreq_Hdeno - LFreq_Hcono
        row['HFreq diff'] = HFreq_Hdeno - HFreq_Hcono
    else:  # eval cono space
        row['cono dev accuracy'] = cono_accuracy
        row['LFreq remove deno'] = LFreq_Hdeno
        row['HFreq remove deno'] = HFreq_Hdeno
        row['LFreq preserve cono'] = LFreq_Hcono
        row['HFreq preserve cono'] = HFreq_Hcono
        row['LFreq diff'] = LFreq_Hcono - LFreq_Hdeno
        row['HFreq diff'] = HFreq_Hcono - HFreq_Hdeno

    for key, val in row.items():
        if isinstance(val, float):
            row[key] = round(val, 4)
    table.append(row)

with open(out_path, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=table[-1].keys(), dialect=csv.excel_tab)
    writer.writeheader()
    writer.writerows(table)
