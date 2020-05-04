from pathlib import Path
from copy import copy
from dataclasses import dataclass
from typing import Tuple, Union, List, Dict, Iterable

import numpy as np
import torch
import editdistance
from scipy.spatial import distance
from tqdm.auto import tqdm

from decomposer import Decomposer, DecomposerConfig
from recomposer import Recomposer, RecomposerConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# pretrained_path = PROJECT_ROOT / 'results/bill topic/pretrained subset/init.pt'
# # pretrained_path = PROJECT_ROOT / 'results/bill topic/pretrained superset/init.pt'
# # pretrained_path = PROJECT_ROOT / 'results/SGNS deno/pretrained super large/init.pt'
# print(f'Loading vocabulary from {pretrained_path}')
# PE = torch.load(pretrained_path)['model']
# PE_embed = PE.embedding.weight.detach().cpu().numpy()


BASE_DIR = Path.home() / 'Research/congressional_adversary/results'
PE = torch.load(BASE_DIR / 'news/validation/pretrained/init.pt')['model']
# PE = torch.load(BASE_DIR / 'news/validation_transformer/pretrained/init.pt')['model']
WTI = PE.word_to_id
ITW = PE.id_to_word
grounding = PE.cono_grounding


def GD(query: str) -> None:
    freq = grounding[WTI[query]]
    ratio = torch.nn.functional.normalize(freq, dim=0, p=1)

    print(query, end='\t')
    for r in ratio.tolist():
        print(round(r, 4), end=', ')
    print(end='\t')

    for f in freq.tolist():
        print(int(f), end=', ')
    print()



PE = PE.embedding.weight.detach().cpu().numpy()
print(f'Vocab size = {len(WTI):,}')


# sub_PE = torch.load(BASE_DIR / 'bill topic/pretrained subset/init.pt')['model']
# sub_PE_WID = sub_PE.word_to_id
# sub_PE_GD = sub_PE.grounding
# del sub_PE


# @dataclass
# class GroundedWord():
#     word: str

# #     def __post_init__(self) -> None:
# #         self.word_id: int = WTI[self.word]
# #         metadata = sub_PE_GD[self.word]
# #         self.freq: int = metadata['freq']
# #         self.R_ratio: float = metadata['R_ratio']
# #         self.majority_deno: int = metadata['majority_deno']

# #         self.PE_neighbors = self.neighbors(PE)

# #     def deno_ground(self, embed, top_k=10):
# #         self.neighbors: List[str] = nearest_neighbors()

#     def __str__(self) -> str:
#         return str(vars(self))


def get_embed(model: Decomposer) -> np.ndarray:
    return model.embedding.weight.detach().cpu().numpy()


def load(
        path: Path,
        match_vocab: bool = False,
        device: str = 'cpu'
        ) -> np.ndarray:
    model = torch.load(path, map_location=device)['model']
    try:
        assert model.word_to_id == WTI
    except AssertionError:
        print(f'Vocabulary mismatch: {path}')
        print(f'Vocab size = {len(model.word_to_id)}')
        if match_vocab:
            raise RuntimeError
        else:
            return None
    return get_embed(model)


def load_decomposers_en_masse(
        in_dirs: Union[Path, List[Path]],
        patterns: Union[str, List[str]]
        ) -> Tuple[Dict[str, np.ndarray], ...]:
    if not isinstance(in_dirs, List):
        in_dirs = [in_dirs, ]
    if not isinstance(patterns, List):
        patterns = [patterns, ]
    checkpoints: List[Path] = []
    for in_dir in in_dirs:
        for pattern in patterns:
            checkpoints += list(in_dir.glob(pattern))
    if len(checkpoints) == 0:
        raise FileNotFoundError('No model with path pattern found at in_dir?')

    models = {
        # 'pretrained superset': load(BASE_DIR / 'bill topic/pretrained superset/init.pt'),
        # 'pretrained subset': load(BASE_DIR / 'bill topic/pretrained subset/init.pt')
    }
    for path in tqdm(checkpoints):
        tqdm.write(f'Loading {path}')
        embed = load(path)
        if embed is None:
            continue
        # name = path.parent.name
        name = path.parent.name + '/' + path.name
        models[name] = embed
    return models


def vec(query: str, embed: np.ndarray) -> np.ndarray:
    try:
        query_id = WTI[query]
    except KeyError:
        raise KeyError(f'Out of vocabulary: {query}')
    return embed[query_id]


def nearest_neighbors(
        query: str,
        embed: np.ndarray,
        top_k: int = 10
        ) -> None:
    query_vec = vec(query, embed)
    # print(f"{query}’s neareset neighbors:")
    distances = [
        distance.cosine(query_vec, neighbor_vec)
        for neighbor_vec in embed]
    neighbor_indices = np.argsort(distances)
    num_neighbors = 0
    for sort_rank, neighbor_id in enumerate(neighbor_indices):
        if num_neighbors == top_k:
            break
        # if query_id == neighbor_id:
        #     continue
        neighbor_word = ITW[neighbor_id]

        if editdistance.eval(query, neighbor_word) < 3:
            continue
        cosine_similarity = 1 - distances[neighbor_id]
        num_neighbors += 1
        print(f'{cosine_similarity:.4f}\t{neighbor_word}')
    print()


def cos_sim(query1: str, query2: str, embed: np.ndarray) -> float:
    v1 = vec(query1, embed)
    v2 = vec(query2, embed)
    return 1 - distance.cosine(v1, v2)


# def cf_cos_sim(
#         q1: str,
#         q2: str,
#         d_embed: np.ndarray,
#         c_embed: np.ndarray
#         ) -> None:
#     deno_cs = cos_sim(q1, q2, d_embed)
#     cono_cs = cos_sim(q1, q2, c_embed)
#     pretrained_cs = cos_sim(q1, q2, PE_embed)
#     print(
#         f'{pretrained_cs:.4f} in pretrained space\n'
#         f'{deno_cs:.4f} in denotation space\n'
#         f'{cono_cs:.4f} in connotation space')
#     print(GroundedWord(q1))
#     print(GroundedWord(q2))


# def tabulate_cos_sim(
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


# @dataclass
# class GroundedWord():
#     word: str

#     def __post_init__(self) -> None:
#         self.word_id: int = PE.word_to_id[self.word]
#         metadata = GD[self.word]
#         self.freq: int = metadata['freq']
#         self.R_ratio: float = metadata['R_ratio']
#         self.majority_deno: int = metadata['majority_deno']
#         # self.majority_deno_id: int = PE.deno_to_id[metadata['majority_deno']]

#     def __str__(self) -> str:
#         return str(vars(self))


# def get_partisan_words(
#         min_skew: float,
#         max_skew: float,
#         min_freq: int,
#         ) -> List[GroundedWord]:
#     samples: List[GroundedWord] = []
#     for word in PE.word_to_id.keys():
#         word = GroundedWord(word)
#         if word.freq >= min_freq and min_skew <= word.R_ratio <= max_skew:
#             samples.append(word)
#     return samples

# min_freq = 0
# chauvinism = get_partisan_words(0.8, 1, min_freq)
# capitalism = get_partisan_words(0.8, 1, min_freq)
# neoliberal_shills = get_partisan_words(0.2, 0.6, min_freq)
# socialism = get_partisan_words(0, 0.2, min_freq)
# bernie_bros = get_partisan_words(0, 0.2, min_freq)

# capitalism: List[GroundedWord] = []
# socialism: List[GroundedWord] = []
# all_words: List[GroundedWord] = []
# for word in PE.word_to_id.keys():
#     word = GroundedWord(word)
#     all_words.append(word)
#     if word.R_ratio < 0.2:  # 0.2:
#         socialism.append(word)
#     # elif word.R_ratio < 0.8:
#     #     cono_bins[1].append(word)
#     elif word.R_ratio > 0.8:  # 0.8:
#         capitalism.append(word)

# print(
#     # f'{len(chauvinism)} chauvinists\n'
#     f'{len(capitalism)} capitalists\n'
#     # f'{len(neoliberal_shills)} neoliberal shills\n'
#     f'{len(socialism)} socialists'
#     # f'{len(bernie_bros)} bernie_bros\n'
# )

# polarized_words = capitalism + socialism
# crisis_words = chauvinism + bernie_bros

# all_words = [GroundedWord(w) for w in PE.word_to_id.keys()]
# capitalism = get_partisan_words(min_skew=0.75, max_skew=1, min_freq=100)
# socialism = get_partisan_words(min_skew=0, max_skew=0.25, min_freq=100)
# polarized_words = capitalism + socialism

# R_words.remove('federal_debt_stood')  # outliers in clustering graphs
# R_words.remove('statements_relating')
# R_words.remove('legislative_days_within')
