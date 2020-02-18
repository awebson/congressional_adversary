import os
import random
from typing import List, Dict, Optional

import torch
from torch import nn
import editdistance
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_completeness_v_measure

from evaluations.euphemism import cherry_words, generic_words

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
sns.set()

DEVICE = 'cpu'
PE = torch.load(
    '../../results/pretrained/init.pt', map_location=DEVICE)['model']
PE.deno_to_id = {val: key for key, val in PE.id_to_deno.items()}
GD = PE.grounding


def load(path):
    stuff = torch.load(path, map_location=DEVICE)['model']
    return stuff.embedding.weight.detach().numpy()


def gather(words):
    word_ids = [PE.word_to_id[w] for w in words]
    freq = [GD[w]['freq'] for w in words]
    skew = [GD[w]['R_ratio'] for w in words]
    maj_deno = [GD[w]['majority_deno'] for w in words]
    return word_ids, freq, skew, maj_deno


def plot(coordinates, words, freq, skew, path):
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


def plot_categorical(coordinates, words, freq, skew, path):
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.scatterplot(
        coordinates[:, 0], coordinates[:, 1],
        hue=skew, palette='muted', hue_norm=(0, 1),
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


def load_en_masse(in_dir, endswith):
    models = {}
    for dirpath, _, filenames in tqdm(os.walk(in_dir)):
        for file in filenames:
            if file.endswith(endswith):
                path = os.path.join(dirpath, file)
                name = path.lstrip(in_dir).replace('/', ' ')
                models[name] = load(path)
    print(*models.keys(), sep='\n')
    return models


def graph_en_masse(
        models,
        out_dir,
        reduction,  # 'PCA', 'TSNE', or 'both'
        word_ids,
        words,
        hues,
        sizes,
        perplexity=None,
        categorical=False):
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
        if not categorical:
            plot(visual, words, sizes, hues,
                 os.path.join(out_dir, f'{model_name}.png'))
        else:
            plot_categorical(
                visual, words, sizes, hues,
                os.path.join(out_dir, f'{model_name}.png'))


def load_recomposer(path):
    stuff = torch.load(path, map_location=DEVICE)['model']
    D_embed = stuff.deno_decomposer.embedding.weight.detach().numpy()
    C_embed = stuff.cono_decomposer.embedding.weight.detach().numpy()
    return D_embed, C_embed

def load_recomposers_en_masse(in_dir, endswith):
    D_models = {
        'pretrained superset': load('../../results/pretrained/init.pt'),
        'pretrained': load('../../results/pretrained bill mentions/init.pt')}
    C_models = {
        'pretrained superset': load('../../results/pretrained/init.pt'),
        'pretrained': load('../../results/pretrained bill mentions/init.pt')}
    for dirpath, _, filenames in os.walk(in_dir):
        for file in filenames:
            if file.endswith(endswith):
                path = os.path.join(dirpath, file)
                name = path.lstrip(in_dir).replace('/', ' ')
                D_embed, C_embed = load_recomposer(path)
                # Brittle Hack
                name = name.split()
                D_name = ' '.join(name[0:2] + name[4:])
                R_name = ' '.join(name[2:])
                D_models[D_name] = D_embed
                C_models[R_name] = C_embed
                print(name)
    return D_models, C_models


def discretize_cono(skew):
    if skew < 0.5:
        return 0
    else:
        return 1


def NN_cluster_ids(embed, query_ids, categorical, top_k=5):
    query_ids = torch.tensor(query_ids, device=DEVICE)
    embed = torch.tensor(embed, device=DEVICE)
    embed = nn.Embedding.from_pretrained(embed, freeze=True)

    query_embed = embed(query_ids)
    top_neighbor_ids = [
        nn.functional.cosine_similarity(
            q.view(1, -1), embed.weight).argsort(descending=True)
        for q in query_embed]

    cluster_labels = []
    true_labels = []
    for query_index, sorted_target_indices in enumerate(top_neighbor_ids):
        query_id = query_ids[query_index].item()
        query_word = PE.id_to_word[query_id]
        num_neighbors = 0
        # if categorical:
        #     query_label = PE.deno_to_id[GD[query_word]['majority_deno']]
        # else:
        #     query_label = discretize_cono(GD[query_word]['R_ratio'])
        query_label = query_index

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


# ch_ids, ch_freq, ch_skew, ch_deno = gather(cherry_words)
# gen_ids, gen_freq, gen_skew, gen_deno = gather(generic_words)
# random_words = [w for w in PE.word_to_id.keys()
#                 if GD[w]['freq'] > 99]
# random_words = random.sample(random_words, 50)
# rand_ids, rand_freq, rand_skew, rand_deno = gather(random_words)

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
