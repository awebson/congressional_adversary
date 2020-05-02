from pathlib import Path
from copy import copy
from dataclasses import dataclass
from typing import Tuple, Union, List, Dict, Iterable

import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

from decomposer import Decomposer, DecomposerConfig
from recomposer import Recomposer, RecomposerConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
pretrained_path = PROJECT_ROOT / 'results/bill topic/pretrained subset/init.pt'
# pretrained_path = PROJECT_ROOT / 'results/bill topic/pretrained superset/init.pt'
# pretrained_path = PROJECT_ROOT / 'results/SGNS deno/pretrained super large/init.pt'
print(f'Loading vocabulary from {pretrained_path}')
PE = torch.load(pretrained_path)['model']
# PE_embed = PE.embedding.weight.detach().cpu().numpy()

GD = PE.grounding


def get_embed(model: Decomposer) -> np.ndarray:
    return model.embedding.weight.detach().cpu().numpy()

PE_embed = get_embed(PE)

def load(
        path: Path,
        check_vocab: bool = True,
        device: str = 'cpu'
        ) -> np.ndarray:
    model = torch.load(path, map_location=device)['model']
    if check_vocab:
        assert model.word_to_id == PE.word_to_id
    return get_embed(model)


def load_en_masse(in_dir: str, endswith: str) -> Dict[str, np.ndarray]:
    import os  # TODO replace with pathlib
    models = {}
    for dirpath, _, filenames in os.walk(in_dir):
        for file in filenames:
            if file.endswith(endswith):
                path = os.path.join(dirpath, file)
                name = path.lstrip(in_dir).replace('/', ' ')
                models[name] = load(path)
    print(*models.keys(), sep='\n')
    return models


def load_recomposer(
        path: Path,
        check_vocab: bool = True,
        device: str = 'cpu'
        ) -> Tuple[np.ndarray, np.ndarray]:
    model = torch.load(path, map_location=device)['model']
    if check_vocab:
        assert model.word_to_id == PE.word_to_id
        assert model.deno_decomposer.word_to_id == PE.word_to_id
        assert model.cono_decomposer.word_to_id == PE.word_to_id
    D_embed = model.deno_decomposer.embedding.weight.detach().numpy()
    C_embed = model.cono_decomposer.embedding.weight.detach().numpy()
    return D_embed, C_embed


def lazy_load_recomposers(
        in_dirs: Union[Path, List[Path]],
        patterns: Union[str, List[str]],
        get_deno_decomposer: bool,
        device: str = 'cpu'
        ) -> Iterable[Decomposer]:
    if not isinstance(in_dirs, List):
        in_dirs = [in_dirs, ]
    if not isinstance(patterns, List):
        patterns = [patterns, ]

    checkpoints: List[Path] = []
    for in_dir in in_dirs:
        for pattern in patterns:
            try:
                checkpoints += list(in_dir.glob(pattern))
            except TypeError:
                raise FileNotFoundError('No model with path pattern found at in_dir?')

    PE1 = torch.load(
        '../../results/pretrained superset/init.pt', map_location=device)['model']
    PE1.name = 'pretrained superset'
    yield PE1
    del PE1
    PE2 = torch.load(
        '../../results/pretrained subset/init.pt', map_location=device)['model']
    PE2.name = 'pretrained subset'
    yield PE2
    del PE2

    serial_number = 0
    for path in tqdm(checkpoints):
        tqdm.write(f'Loading {path}')
        cucumbers = torch.load(path, map_location=device)
        recomp = cucumbers['model']
        config = cucumbers['config']

        if get_deno_decomposer:
            decomp = recomp.deno_decomposer
        else:
            decomp = recomp.cono_decomposer

        assert decomp.word_to_id == PE.word_to_id
        decomp.config = config
        decomp.name = f'M{serial_number}'
        decomp.stem = path.stem
        decomp.eval()
        yield decomp
        serial_number += 1


def load_recomposers_en_masse(
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

    D_models = {
        'pretrained superset': load(PROJECT_ROOT / 'results/pretrained superset/init.pt'),
        'pretrained subset': load(PROJECT_ROOT / 'results/pretrained subset/init.pt')}
    C_models = copy(D_models)
    for path in tqdm(checkpoints):
        tqdm.write(f'Loading {path}')
        D_embed, C_embed = load_recomposer(path)
        # Brittle convenience
        name = path.parent.name.split()
        D_name = ' '.join(name[0:2] + name[4:])
        R_name = ' '.join(name[2:])
        D_models[D_name] = D_embed
        C_models[R_name] = C_embed
    return D_models, C_models


def vec(query: str, embed: np.ndarray) -> np.ndarray:
    try:
        query_id = PE.word_to_id[query]
    except KeyError:
        raise KeyError(f'Out of vocabulary: {query}')
    return embed[query_id]


def cos_sim(query1: str, query2: str, embed: np.ndarray) -> float:
    v1 = vec(query1, embed)
    v2 = vec(query2, embed)
    return 1 - distance.cosine(v1, v2)


def cf_cos_sim(
        q1: str,
        q2: str,
        d_embed: np.ndarray,
        c_embed: np.ndarray
        ) -> None:
    deno_cs = cos_sim(q1, q2, d_embed)
    cono_cs = cos_sim(q1, q2, c_embed)
    pretrained_cs = cos_sim(q1, q2, PE_embed)
    print(
        f'{pretrained_cs:.4f} in pretrained space\n'
        f'{deno_cs:.4f} in denotation space\n'
        f'{cono_cs:.4f} in connotation space')
    print(GroundedWord(q1))
    print(GroundedWord(q2))


def tabulate_cos_sim(
        pairs: List[Tuple[str, str]],
        d_embed: np.ndarray,
        c_embed: np.ndarray
        ) -> None:
    print('w1', 'w2', 'pretrained', 'deno', 'cono', 'diff', sep='\t')
    for q1, q2 in pairs:
        pretrained_cs = round(cos_sim(q1, q2, PE_embed), 4)
        deno_cs = round(cos_sim(q1, q2, d_embed), 4)
        cono_cs = round(cos_sim(q1, q2, c_embed), 4)
        diff = round(deno_cs - cono_cs, 4)
        print(q1, q2, pretrained_cs, deno_cs, cono_cs, diff, sep='\t')


def nearest_neighbors(
        query: Union[str, np.ndarray],
        embed: np.ndarray,
        top_k: int = 10
        ) -> None:
    if isinstance(query, str):
        query_vec = vec(query, embed)
        print(f"{query}'s neareset neighbors:")
    else:
        query_vec = query
    distances = [
        distance.cosine(query_vec, neighbor_vec)
        for neighbor_vec in embed]
    neighbors = np.argsort(distances)
    for ranking in range(0, top_k + 1):
        word_id = neighbors[ranking]
        word = PE.id_to_word[word_id]
        cosine_similarity = 1 - distances[word_id]
        print(f'{cosine_similarity:.4f}\t{word}')
    # cos_sim = nn.functional.cosine_similarity(
    #     query_vec.view(1, -1), self.embedding)
    # top_neighbor_ids = cos_sim.argsort(descending=True)[1:top_k + 1]
    # for neighbor_id in top_neighbor_ids:
    #     neighbor_id = neighbor_id.item()
    #     neighbor_word = self.id_to_word[neighbor_id]
    #     sim = cos_sim[neighbor_id]
    #     print(f'{sim:.4f}\t{neighbor_word}')
    print()


@dataclass
class GroundedWord():
    word: str

    def __post_init__(self) -> None:
        self.word_id: int = PE.word_to_id[self.word]
        metadata = GD[self.word]
        self.freq: int = metadata['freq']
        self.R_ratio: float = metadata['R_ratio']
        self.majority_deno: int = metadata['majority_deno']
        # self.majority_deno_id: int = PE.deno_to_id[metadata['majority_deno']]

    def __str__(self) -> str:
        return str(vars(self))


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


capitalism: List[GroundedWord] = []
socialism: List[GroundedWord] = []
all_words: List[GroundedWord] = []
for word in PE.word_to_id.keys():
    word = GroundedWord(word)
    all_words.append(word)
    if word.R_ratio < 0.2:  # 0.2:
        socialism.append(word)
    # elif word.R_ratio < 0.8:
    #     cono_bins[1].append(word)
    elif word.R_ratio > 0.8:  # 0.8:
        capitalism.append(word)

print(
    # f'{len(chauvinism)} chauvinists\n'
    f'{len(capitalism)} capitalists\n'
    # f'{len(neoliberal_shills)} neoliberal shills\n'
    f'{len(socialism)} socialists'
    # f'{len(bernie_bros)} bernie_bros\n'
)

polarized_words = capitalism + socialism
# crisis_words = chauvinism + bernie_bros

# all_words = [GroundedWord(w) for w in PE.word_to_id.keys()]
# capitalism = get_partisan_words(min_skew=0.75, max_skew=1, min_freq=100)
# socialism = get_partisan_words(min_skew=0, max_skew=0.25, min_freq=100)
# polarized_words = capitalism + socialism

# R_words.remove('federal_debt_stood')  # outliers in clustering graphs
# R_words.remove('statements_relating')
# R_words.remove('legislative_days_within')
