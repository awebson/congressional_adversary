import os
import random
from typing import Tuple, Union, List, Dict, Optional

import torch
from torch import nn
import editdistance
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from decomposer import Decomposer, DecomposerConfig
from evaluations.euphemism import cherry_words

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
sns.set()

DEVICE = 'cpu'


# HACK replace with os.path.dirname(os.path.dirname(__file__))
pretrained_path = os.path.expanduser(
    '~/Research/congressional_adversary/results/pretrained superset/init.pt')

PE = torch.load(pretrained_path, map_location=DEVICE)['model']
PE.deno_to_id = {val: key for key, val in PE.id_to_deno.items()}
GD = PE.grounding

def gather(
        words: List[str]
        ) -> Tuple[List[int], List[int], List[float], List[str]]:
    word_ids = [PE.word_to_id[w] for w in words]
    freq = [GD[w]['freq'] for w in words]
    skew = [GD[w]['R_ratio'] for w in words]
    maj_deno = [GD[w]['majority_deno'] for w in words]
    return word_ids, freq, skew, maj_deno


def plot(
        coordinates: np.ndarray,
        words: List[str],
        freq: List[int],
        skew: List[float],
        path: str
        ) -> None:
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.scatterplot(
        coordinates[:, 0], coordinates[:, 1],
        hue=skew, palette='coolwarm',  # hue_norm=(0, 1),
        size=freq, sizes=(100, 1000),
        legend=None, ax=ax)
    for coord, word in zip(coordinates, words):
        ax.annotate(word, coord, fontsize=12)
    with open(path, 'wb') as file:
        fig.savefig(file, dpi=300)
    plt.close(fig)


def plot_categorical(
        coordinates: np.ndarray,
        words: List[str],
        freq: List[int],
        categories: List[int],
        path: str
        ) -> None:
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.scatterplot(
        coordinates[:, 0], coordinates[:, 1],
        hue=categories, palette='muted', hue_norm=(0, 1),
        size=freq, sizes=(100, 1000),
        legend='brief', ax=ax)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), ncol=1)
    for coord, word in zip(coordinates, words):
        ax.annotate(word, coord, fontsize=12)
    with open(path, 'wb') as file:
        fig.savefig(file, dpi=300)
    plt.close(fig)


def graph_en_masse(
        models: Dict[str, np.ndarray],
        out_dir: str,
        reduction: str,  # 'PCA', 'TSNE', or 'both'
        word_ids: List[int],
        words: List[str],
        hues: Union[List[float], List[int]],
        sizes: List[int],
        perplexity: Optional[int] = None,
        categorical: bool = False
        ) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for model_name, embed in tqdm(models.items()):
        space = embed[word_ids]
        if reduction == 'PCA':
            visual = PCA(n_components=2).fit_transform(space)
        elif reduction == 'TSNE':
            assert perplexity is not None
            visual = TSNE(
                perplexity=perplexity, learning_rate=10,
                n_iter=5000, n_iter_without_progress=1000).fit_transform(space)
        elif reduction == 'both':
            assert perplexity is not None
            space = PCA(n_components=30).fit_transform(space)
            visual = TSNE(
                perplexity=perplexity, learning_rate=10,
                n_iter=5000, n_iter_without_progress=1000).fit_transform(space)
        else:
            raise ValueError('unknown dimension reduction method')
        if categorical:
            plot_categorical(
                visual, words, sizes, hues,
                os.path.join(out_dir, f'{model_name}.png'))
        else:
            plot(visual, words, sizes, hues,
                 os.path.join(out_dir, f'{model_name}.png'))


def discretize_cono(skew: float) -> int:
    if skew < 0.5:
        return 0
    else:
        return 1


def NN_cluster_ids(
        embed: np.ndarray,
        query_ids: List[int],
        categorical: bool,
        top_k: int = 5
        ) -> Tuple[List[int], List[int]]:
    query_ids = torch.tensor(query_ids, device=DEVICE)
    embed = torch.tensor(embed, device=DEVICE)
    embed = nn.Embedding.from_pretrained(embed, freeze=True)
    with torch.no_grad():
        query_vectors = embed(query_ids)
        cos_sim = nn.functional.cosine_similarity(
            query_vectors.unsqueeze(1),
            embed.weight.unsqueeze(0),
            dim=2)
        # add some buffer top_k for exluding low edit distance neighbors
        cos_sim, neighbor_ids = cos_sim.topk(k=top_k + 10, dim=-1)

    cluster_labels = []
    true_labels = []
    for query_index, sorted_target_indices in enumerate(neighbor_ids):
        query_id = query_ids[query_index].item()
        query_word = PE.id_to_word[query_id]
        query_label = query_index
        num_neighbors = 0
        for sort_rank, target_id in enumerate(sorted_target_indices):
            target_id = target_id.item()
            if num_neighbors == top_k:
                break
            if query_id == target_id:
                continue
            # target_id = target_ids[target_index]  # target is always all embed
            target_word = PE.id_to_word[target_id]
            if editdistance.eval(query_word, target_word) < 3:
                continue
            num_neighbors += 1

            if categorical:
                neighbor_label = PE.deno_to_id[GD[target_word]['majority_deno']]
            else:
                neighbor_label = discretize_cono(GD[target_word]['R_ratio'])
            cluster_labels.append(query_label)
            true_labels.append(neighbor_label)
    return cluster_labels, true_labels


cherry_ids, cherry_freq, cherry_skew, cherry_deno = gather(cherry_words)
# gen_ids, gen_freq, gen_skew, gen_deno = gather(generic_words)
# random_words = [w for w in PE.word_to_id.keys()
#                 if GD[w]['freq'] > 99]
# random_words = random.sample(random_words, 50)
# rand_ids, rand_freq, rand_skew, rand_deno = gather(random_words)
# print(f'{len(GOP_ids)} capitalists\n'
#       f'{len(Dem_ids)} socialists\n'
#       f'{len(neutral_ids)} neoliberal shills\n')

R_words = [w for w in PE.word_to_id.keys()
           if GD[w]['freq'] > 99 and GD[w]['R_ratio'] > 0.75]
R_words.remove('federal_debt_stood')  # outliers in clustering graphs
R_words.remove('statements_relating')
R_words.remove('legislative_days_within')
# print(len(R_words))
# GOP_words = random.sample(GOP_words, 50)
R_ids, R_freq, R_skew, R_deno = gather(R_words)

# D_words = [w for w in PE.word_to_id.keys()
#            if GD[w]['freq'] > 99 and GD[w]['R_ratio'] < 0.25]

D_words = [
    'war_in_iraq', 'unemployed', 'detainees', 'solar',
    'wealthiest', 'minorities', 'gun_violence',
    'amtrak', 'unemployment_benefits',
    'citizens_united', 'mayors', 'prosecutor', 'working_families',
    'cpsc', 'sexual_assault',
    'affordable_housing', 'vietnam_veterans', 'drug_companies', 'handguns',
    'hungry', 'college_education',
    'main_street', 'trauma', 'simon', 'pandemic',
    'reagan_administration', 'guns',
    'million_jobs', 'airline_industry', 'mergers', 'blacks',
    'industrial_base', 'unemployment_insurance',
    'vacancies', 'trade_deficit', 'lost_their_jobs', 'food_safety',
    'darfur', 'trains', 'deportation', 'credit_cards',
    'surface_transportation', 'solar_energy', 'ecosystems', 'layoffs',
    'wall_street', 'steelworkers', 'puerto_rico', 'hunger',
    'child_support', 'naacp', 'domestic_violence', 'seaports',
    'hate_crimes', 'underfunded', 'registrants', 'sanctuary',
    'coastal_zone_management', 'vermonters', 'automakers',
    'violence_against_women', 'unemployment_rate',
    'select_committee_on_indian_affairs', 'judicial_nominees',
    'school_construction', 'clarence_mitchell', 'confidential',
    'domain_name', 'community_development', 'pell_grant', 'asylum', 'vawa',
    'somalia', 'african_american', 'traders', 'jersey', 'fdic', 'shameful',
    'homelessness', 'african_americans', 'payroll_tax', ]
#     'retraining', 'unemployed_workers', 'the_disclose_act', 'baltimore',
#     'assault_weapons', 'credit_card', 'the_patriot_act', 'young_woman',
#     'trades', 'aye', 'poisoning', 'police_officers', 'mammal', 'toys',
#     'whistleblowers', 'north_dakota', 'californias', 'computer_crime',
#     'explosives', 'fast_track', 'bus', 'redlining', 'seclusion', 'gender',
#     'hawaiian', 'pay_discrimination', 'ledbetter', 'phd', 'supra', 'baggage',
#     'las_vegas', 'the_voting_rights_act', 'enron', 'richest', 'vra', 'chip',
#     'tax_break', 'the_usa_patriot_act', 'advance_notice', 'derivatives',
#     'the_patients_bill_of_rights', 'shelf', 'divestment', 'sa',
#     'submitted_an_amendment', 'bill_hr', 'first_responders',
#     'unemployment_compensation', 'tax_breaks', 'carbon',
#     'college_cost_reduction', 'clean_energy', 'waives',
#     'unregulated', 'taa', 'truman', 'lesbian', 'coupons',
#     'large_numbers', 'anonymous', 'whites', 'logging']

# print(len(D_words))
D_words = random.sample(D_words, 50)
D_ids, D_freq, D_skew, D_deno = gather(D_words)

J_words = D_words + R_words
J_ids = D_ids + R_ids
J_freq = D_freq + R_freq
J_skew = D_skew + R_skew
J_deno = D_deno + R_deno
J_cono = [0 if skew < 0.5 else 1 for skew in J_skew]