import csv
import pickle
import random
import warnings
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.nn.utils import rnn

from decomposer import Decomposer, DecomposerConfig, LabeledSentences
from recomposer import Recomposer, RecomposerConfig
from evaluations.helpers import lazy_load_recomposers, polarized_words, all_words

from evaluations.helpers import cos_sim, PE_embed, get_embed
from evaluations.euphemism import cherry_pairs

random.seed(42)

warnings.simplefilter('ignore')
DEVICE = 'cuda:0'
EVAL_DENO_SPACE = False
# out_path = '../../analysis/deno_space_3bin.tsv'
out_path = '../../analysis/cono_space_3bin.tsv'

decomposers = lazy_load_recomposers(
    in_dirs=[Path('../../results/recomposer'), Path('../../results/sans recomposer')],
    patterns=['*/epoch10.pt', '*/epoch30.pt', '*/epoch50.pt', '*/epoch80.pt', '*/epoch100.pt'],
    get_deno_decomposer=EVAL_DENO_SPACE,
    device=DEVICE)

# # Debug Path
# out_path = '../../analysis/debug.tsv'
# decomposers = lazy_load_recomposers(
#     in_dirs=Path('../../results/sans recomposer'),
#     patterns='*/epoch50.pt',
#     get_deno_decomposer=EVAL_DENO_SPACE,
#     device=DEVICE)


polarized_ids = torch.tensor([w.word_id for w in polarized_words], device=DEVICE)
HF_polarized_ids = torch.tensor([w.word_id for w in polarized_words if w.freq > 99], device=DEVICE)
# crisis_ids = torch.tensor([w.word_id for w in crisis_words], device=DEVICE)

random_words = random.sample(all_words, 1000)
random_ids = torch.tensor([w.word_id for w in random_words], device=DEVICE)

HF_random_ids = [w.word_id for w in all_words if w.freq > 99]
HF_random_ids = random.sample(HF_random_ids, 1000)
HF_random_ids = torch.tensor(HF_random_ids, device=DEVICE)


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
for model in decomposers:
    HFreq_Hdeno = model.NN_cluster_homogeneity(HF_polarized_ids, eval_deno=True, top_k=5)
    HFreq_Hcono = model.NN_cluster_homogeneity(HF_polarized_ids, eval_deno=False, top_k=5)
    LFreq_Hdeno = model.NN_cluster_homogeneity(polarized_ids, eval_deno=True, top_k=5)
    LFreq_Hcono = model.NN_cluster_homogeneity(polarized_ids, eval_deno=False, top_k=5)

    R_HFreq_Hdeno = model.NN_cluster_homogeneity(HF_random_ids, eval_deno=True, top_k=5)
    R_HFreq_Hcono = model.NN_cluster_homogeneity(HF_random_ids, eval_deno=False, top_k=5)
    R_LFreq_Hdeno = model.NN_cluster_homogeneity(random_ids, eval_deno=True, top_k=5)
    R_LFreq_Hcono = model.NN_cluster_homogeneity(random_ids, eval_deno=False, top_k=5)

    deno_accuracy, cono_accuracy = model.accuracy(dev_seq, dev_deno_labels, dev_cono_labels)
    decomposed_embed = get_embed(model)

    if 'pretrained' in model.name:
        row = {'model_name': model.name}
    else:
        row = {
            'model_name': model.name,
            'epoch': model.stem.lstrip('epoch'),
            r'delta': model.delta,
            r'gamma': model.gamma,
            r'delta / gamma': model.delta / model.gamma,
            r'rho': model.config.recomposer_rho,
            'dropout': model.config.dropout_p}

    if EVAL_DENO_SPACE:
        row['deno dev accuracy'] = deno_accuracy
        row['cono dev accuracy'] = cono_accuracy
        row['preserve deno'] = LFreq_Hdeno
        row['remove cono'] = LFreq_Hcono
        row['homo diff'] = LFreq_Hdeno - LFreq_Hcono
        row['HF preserve deno'] = HFreq_Hdeno
        row['HF remove cono'] = HFreq_Hcono
        row['HF homo diff'] = HFreq_Hdeno - HFreq_Hcono

        row['R preserve deno'] = R_LFreq_Hdeno
        row['R remove cono'] = R_LFreq_Hcono
        row['R homo diff'] = R_LFreq_Hdeno - R_LFreq_Hcono
        row['R HF preserve deno'] = R_HFreq_Hdeno
        row['R HF remove cono'] = R_HFreq_Hcono
        row['R HF homo diff'] = R_HFreq_Hdeno - R_HFreq_Hcono

        diffs = []
        for w1, w2 in cherry_pairs:
            pair = '_' + w1[:6] + '_' + w2[:6]
            pretrained_cs = cos_sim(w1, w2, PE_embed)
            deno_cs = cos_sim(w1, w2, decomposed_embed)
            row['PSim' + pair] = pretrained_cs
            row['DSim' + pair] = deno_cs
            row['Diff' + pair] = deno_cs - pretrained_cs
            diffs.append(deno_cs - pretrained_cs)
        row['mean diff'] = np.mean(diffs)

    else:  # eval cono space
        row['deno dev accuracy'] = deno_accuracy
        row['cono dev accuracy'] = cono_accuracy
        row['remove deno'] = LFreq_Hdeno
        row['preserve cono'] = LFreq_Hcono
        row['homo diff'] = LFreq_Hcono - LFreq_Hdeno
        row['HF remove deno'] = HFreq_Hdeno
        row['HF preserve cono'] = HFreq_Hcono
        row['HF homo diff'] = HFreq_Hcono - HFreq_Hdeno

        row['R remove deno'] = R_LFreq_Hdeno
        row['R preserve cono'] = R_LFreq_Hcono
        row['R homo diff'] = R_LFreq_Hdeno - R_LFreq_Hcono
        row['R HF remove deno'] = R_HFreq_Hdeno
        row['R HF preserve cono'] = R_HFreq_Hcono
        row['R HF homo diff'] = R_HFreq_Hdeno - R_HFreq_Hcono

        diffs = []
        for w1, w2 in cherry_pairs:
            pair = '_' + w1[:6] + '_' + w2[:6]
            pretrained_cs = cos_sim(w1, w2, PE_embed)
            cono_cs = cos_sim(w1, w2, decomposed_embed)
            row['PSim' + pair] = pretrained_cs
            row['CSim' + pair] = cono_cs
            row['Diff' + pair] = pretrained_cs - cono_cs
            diffs.append(pretrained_cs - cono_cs)
        row['mean diff'] = np.mean(diffs)

    for key, val in row.items():
        if isinstance(val, float):
            row[key] = round(val, 4)
    table.append(row)

with open(out_path, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=table[-1].keys(), dialect=csv.excel_tab)
    writer.writeheader()
    writer.writerows(table)
