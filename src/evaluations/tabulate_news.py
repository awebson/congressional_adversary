import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Set, Tuple, Union, List, Dict, Iterable

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from decomposer import Decomposer, DecomposerConfig
from recomposer import Recomposer, RecomposerConfig


def main() -> None:
    in_dir = Path('../../results/search')
    patterns = ['*/epoch*.pt']  #,'duplicate/*/epoch*.pt']
    out_path = in_dir / 'debug.tsv'
    random_path = Path('../../data/ellie/rand_sample.hp.txt')
    test_path = Path('../../data/ellie/partisan_sample.hp.txt')
    device = torch.device('cuda:0')

    # tensorboard = SummaryWriter(log_dir=log_dir)
    checkpoints: List[Path] = []
    for pattern in patterns:
        checkpoints += list(in_dir.glob(pattern))
    if len(checkpoints) == 0:
        raise FileNotFoundError(f'No model with path pattern found at {in_dir}?')

    with open(random_path) as file:
        random_words = [word.strip() for word in file]
    with open(test_path) as file:
        test_words = [word.strip() for word in file]

    wid = None
    table: List[Dict] = []
    for path in tqdm(checkpoints):
        tqdm.write(f'Loading {path}')
        payload = torch.load(path, map_location=device)
        model = payload['model']
        config = payload['config']
        row = {
            'path': path.parent.name + '/' + path.name,  # path.parent.name
            'deduplicated': 'duplicate' not in str(path),
            'D_delta': config.deno_delta,
            'D_gamma': config.deno_gamma,
            'C_delta': config.cono_delta,
            'C_gamma': config.cono_gamma,
            'rho': config.recomposer_rho,
            'max_adversary_loss': config.max_adversary_loss,
            'batch size': config.batch_size,
            'learning rate': config.learning_rate,
        }

        D_model = model.deno_decomposer
        C_model = model.cono_decomposer

        # HACK
        if len(table) == 0:
            row['path'] = 'pretrained'
            D_model.embedding = D_model.pretrained_embed
            C_model.embedding = C_model.pretrained_embed


        if wid is not None:
            assert wid == model.word_to_id
        else:
            wid = model.word_to_id
            # random = torch.tensor([wid[word] for word in random_words], device=device)
            # test = torch.tensor([wid[word] for word in test_words], device=device)
            dev = torch.cat([D_model.liberal_ids, D_model.neutral_ids, D_model.conservative_ids])

            # def word_id_tensor(words: List[str]) -> torch.Tensor:
            #     wids = []
            #     skipped = []
            #     for word in words:
            #         try:
            #             wids.append(model.word_to_id[word])
            #         except KeyError:
            #             skipped.append(word)
            #             continue
            #     if skipped:
            #         print(f'{len(skipped)} OOV: {skipped}')
            #     return torch.tensor(wids, device=device)

            # random = word_id_tensor(random)
            # test = word_id_tensor(test)


            # Initailize denotation grounding
            # def pretrained_neighbors(
            #         query_ids: torch.Tensor,
            #         top_k: int = 10
            #         ) -> Dict[int, Set[int]]:
            #     import editdistance
            #     import torch.nn.functional as F

            #     deno_grounding: Dict[int, Set[int]] = {}
            #     for qid in query_ids:
            #         qv = D_model.pretrained_embed(qid)
            #         qid = qid.item()
            #         qw = model.id_to_word[qid]
            #         cos_sim = F.cosine_similarity(qv.unsqueeze(0), D_model.pretrained_embed.weight)
            #         cos_sim, neighbor_ids = cos_sim.topk(k=top_k + 5, dim=-1)
            #         neighbor_ids = [
            #             nid for nid in neighbor_ids.tolist()
            #             if editdistance.eval(qw, model.id_to_word[nid]) > 3]
            #         deno_grounding[qid] = set(neighbor_ids[:top_k])
            #     return deno_grounding

            # # rand_deno = pretrained_neighbors(random)
            # test_deno = pretrained_neighbors(test)

        # D_model.deno_grounding.update(rand_deno)
        # D_model.deno_grounding.update(test_deno)
        # C_model.deno_grounding.update(rand_deno)
        # C_model.deno_grounding.update(test_deno)


        DH_rand, CKL_rand, CH_rand = D_model.homogeneity(random)
        DH_dev, CKL_dev, CH_dev = D_model.homogeneity(dev)
        DH_test, CKL_test, CH_test = D_model.homogeneity(test)

        D_rand_diff = DH_rand - CH_rand
        D_dev_diff = DH_dev - CH_dev
        D_test_diff = DH_test - CH_test
        row['DVec rand homo diff'] = D_rand_diff
        row['DVec dev homo diff'] = D_dev_diff
        row['DVec test homo diff'] = D_test_diff

        DH_rand, CKL_rand, CH_rand = C_model.homogeneity(random)
        DH_dev, CKL_dev, CH_dev = C_model.homogeneity(dev)
        DH_test, CKL_test, CH_test = C_model.homogeneity(test)

        C_rand_diff = CH_rand - DH_rand
        C_dev_diff = CH_dev - DH_dev
        C_test_diff = CH_test - DH_test
        row['CVec rand homo diff'] = C_rand_diff
        row['CVec dev homo diff'] = C_dev_diff
        row['CVec test homo diff'] = C_test_diff

        row['mean rand homo diff'] = np.mean([D_rand_diff, C_rand_diff])
        row['mean dev homo diff'] = np.mean([D_dev_diff, C_dev_diff])
        row['mean test homo diff'] = np.mean([D_test_diff, C_test_diff])
        table.append(row)

        break  # HACK

    with open(out_path, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=table[-1].keys(), dialect=csv.excel_tab)
        writer.writeheader()
        writer.writerows(table)


if __name__ == '__main__':
    main()
