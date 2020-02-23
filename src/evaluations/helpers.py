import os
from copy import copy
from typing import Tuple, Union, List, Dict, Iterable

import numpy as np
import torch
from tqdm import tqdm

from decomposer import Decomposer, DecomposerConfig
from recomposer import Recomposer, RecomposerConfig


# HACK replace with os.path.dirname(os.path.dirname(__file__))
pretrained_path = os.path.expanduser(
    '~/Research/congressional_adversary/results/pretrained superset/init.pt')
VOCAB = torch.load(pretrained_path)['model'].word_to_id


def get_embed(model: Decomposer) -> np.ndarray:
    return model.embedding.weight.detach().numpy()


def load(
        path: str,
        check_vocab: bool = True,
        device: str = 'cpu'
        ) -> np.ndarray:
    model = torch.load(path, map_location=device)['model']
    if check_vocab:
        assert model.word_to_id == VOCAB
    return get_embed(model)


def load_en_masse(in_dir: str, endswith: str) -> Dict[str, np.ndarray]:
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
        in_dirs: Union[str, List[str]],
        suffixes: Union[str, List[str]],
        get_deno_decomposer: bool,
        device: str = 'cpu'
        ) -> Iterable[Decomposer]:
    if not isinstance(in_dirs, List):
        in_dirs = [in_dirs, ]
    if not isinstance(suffixes, List):
        suffixes = [suffixes, ]
    checkpoints = []
    for in_dir in in_dirs:
        for dirpath, _, filenames in os.walk(in_dir):
            for file in filenames:
                for suffix in suffixes:
                    if file.endswith(suffix):
                        checkpoints.append(os.path.join(dirpath, file))
    if len(checkpoints) == 0:
        raise FileNotFoundError('No model with path suffix found at in_dir?')

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

        assert decomp.word_to_id == VOCAB
        decomp.config = config
        decomp.suffix = path.split('/')[-1]  # HACK
        decomp.name = f'M{serial_number}'
        decomp.eval()
        yield decomp
        serial_number += 1


def load_recomposers_en_masse(
        in_dirs: Union[str, List[str]],
        suffixes: Union[str, List[str]]
        ) -> Tuple[Dict[str, np.ndarray], ...]:
    if not isinstance(in_dirs, List):
        in_dirs = [in_dirs, ]
    if not isinstance(suffixes, List):
        suffixes = [suffixes, ]
    checkpoints = []
    for in_dir in in_dirs:
        for dirpath, _, filenames in os.walk(in_dir):
            for file in filenames:
                for suffix in suffixes:
                    if file.endswith(suffix):
                        checkpoints.append(os.path.join(dirpath, file))
    if len(checkpoints) == 0:
        raise FileNotFoundError('No model with path suffix found at in_dir?')

    D_models = {
        'pretrained superset': load('../../results/pretrained superset/init.pt'),
        'pretrained subset': load('../../results/pretrained subset/init.pt')}
    C_models = {
        'pretrained superset': load('../../results/pretrained superset/init.pt'),
        'pretrained subset': load('../../results/pretrained subset/init.pt')}
    for path in tqdm(checkpoints):
        tqdm.write(f'Loading {path}')
        name = path.lstrip(in_dir).replace('/', ' ')
        D_embed, C_embed = load_recomposer(path)
        # Brittle convenience
        name = name.split()
        D_name = ' '.join(name[0:2] + name[4:])
        R_name = ' '.join(name[2:])
        D_models[D_name] = D_embed
        C_models[R_name] = C_embed
    return D_models, C_models
