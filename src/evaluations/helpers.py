from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Union, List, Dict, Iterable

import numpy as np
import torch
from tqdm import tqdm

from decomposer import Decomposer, DecomposerConfig
from recomposer import Recomposer, RecomposerConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
pretrained_path = PROJECT_ROOT / 'results/pretrained superset/init.pt'
PE = torch.load(pretrained_path)['model']
GD = PE.grounding


def get_embed(model: Decomposer) -> np.ndarray:
    return model.embedding.weight.detach().numpy()


def load(
        path: str,
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
        path: str,
        check_vocab: bool = True,
        device: str = 'cpu'
        ) -> Tuple[np.ndarray, np.ndarray]:
    stuff = torch.load(path, map_location=device)['model']
    D_embed = stuff.deno_decomposer.embedding.weight.detach().numpy()
    C_embed = stuff.cono_decomposer.embedding.weight.detach().numpy()
    return D_embed, C_embed


# def temp_config_parser(path: str) -> Dict[str, str]:
#     config = {}
#     with open(path) as file:
#         for line in file:
#             if '=' in line:
#                 line = line.split('=')
#                 config[line[0].strip()] = line[1].strip()
#     return config


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
        'pretrained superset': load('../../results/pretrained superset/init.pt'),
        'pretrained subset': load('../../results/pretrained subset/init.pt')}
    C_models = {
        'pretrained superset': load('../../results/pretrained superset/init.pt'),
        'pretrained subset': load('../../results/pretrained subset/init.pt')}
    for path in tqdm(checkpoints):
        tqdm.write(f'Loading {path}')
        D_embed, C_embed = load_recomposer(path)
        # Brittle convenience
        name = path.parent.name.parts
        D_name = ' '.join(name[0:2] + name[4:])
        R_name = ' '.join(name[2:])
        D_models[D_name] = D_embed
        C_models[R_name] = C_embed
    return D_models, C_models


@dataclass
class GroundedWord():
    word: str

    def __post_init__(self) -> None:
        self.word_id: int = PE.word_to_id[self.word]
        metadata = GD[self.word]
        self.freq: int = metadata['freq']
        self.R_ratio: float = metadata['R_ratio']
        self.majority_deno_id: int = PE.deno_to_id[metadata['majority_deno']]

def get_partisan_words(
        min_skew: float,
        max_skew: float,
        min_freq: int,
        ) -> List[GroundedWord]:
    samples: List[GroundedWord] = []
    for word in PE.word_to_id.keys():
        word = GroundedWord(word)
        if word.freq >= min_freq and min_skew < word.R_ratio < max_skew:
            samples.append(word)
    return samples


capitalism = get_partisan_words(min_skew=0.75, max_skew=1, min_freq=100)
socialism = get_partisan_words(min_skew=0, max_skew=0.25, min_freq=100)
polarized_words = capitalism + socialism

print(
    f'{len(capitalism)} capitalists\n'
    f'{len(socialism)} socialists\n'
    # f'{len(neutral_ids)} neoliberal shills\n'
)

# R_words.remove('federal_debt_stood')  # outliers in clustering graphs
# R_words.remove('statements_relating')
# R_words.remove('legislative_days_within')
