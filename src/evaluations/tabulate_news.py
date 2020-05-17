import csv
from pathlib import Path
from typing import Set, Tuple, List, Dict

import numpy as np
import torch
from tqdm import tqdm

from decomposer import Decomposer, DecomposerConfig
from recomposer import Recomposer, RecomposerConfig


def evaluate(word_ids, suffix, D_model, C_model) -> Dict[str, float]:
    row = {}
    DS_Hdeno, _, DS_Hcono = D_model.homogeneity(word_ids)
    row['DS Hdeno'] = DS_Hdeno
    row['DS Hcono'] = DS_Hcono
    row['IntraDS Hd - Hc'] = DS_Hdeno - DS_Hcono

    CS_Hdeno, _, CS_Hcono = C_model.homogeneity(word_ids)
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


def main() -> None:
    in_dir = Path('../../results/search')
    patterns = ['*/epoch*.pt']  # 'duplicate/*/epoch*.pt']
    out_path = in_dir / 'diagonal.tsv'
    random_path = Path('../../data/ellie/rand_sample.hp.txt')
    test_path = Path('../../data/ellie/partisan_sample.hp.txt')
    device = torch.device('cuda:0')
    debug = False

    # tensorboard = SummaryWriter(log_dir=log_dir)
    checkpoints: List[Path] = []
    for pattern in patterns:
        checkpoints += list(in_dir.glob(pattern))
    if len(checkpoints) == 0:
        raise FileNotFoundError(f'No model with path pattern found at {in_dir}?')

    # with open(random_path) as file:
    #     random_words = [word.strip() for word in file]
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

        if wid is not None:
            assert wid == model.word_to_id == D_model.word_to_id == C_model.word_to_id
        else:
            wid = model.word_to_id
            # random = torch.tensor([wid[word] for word in random_words], device=device)
            test_ids = torch.tensor([wid[word] for word in test_words], device=device)

            # For OOV
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
            def pretrained_neighbors(
                    query_ids: torch.Tensor,
                    top_k: int = 10
                    ) -> Dict[int, Set[int]]:
                import editdistance
                import torch.nn.functional as F

                deno_grounding: Dict[int, Set[int]] = {}
                for qid in query_ids:
                    qv = D_model.pretrained_embed(qid)
                    qid = qid.item()
                    qw = model.id_to_word[qid]
                    cos_sim = F.cosine_similarity(qv.unsqueeze(0), D_model.pretrained_embed.weight)
                    cos_sim, neighbor_ids = cos_sim.topk(k=top_k + 5, dim=-1)
                    neighbor_ids = [
                        nid for nid in neighbor_ids.tolist()
                        if editdistance.eval(qw, model.id_to_word[nid]) > 3]
                    deno_grounding[qid] = set(neighbor_ids[:top_k])
                return deno_grounding

            # rand_deno = pretrained_neighbors(random)
            test_deno = pretrained_neighbors(test_ids)

        dev_ids = torch.cat([D_model.liberal_ids, D_model.neutral_ids, D_model.conservative_ids])

        D_model.deno_grounding.update(test_deno)
        C_model.deno_grounding.update(test_deno)
        # D_model.deno_grounding.update(rand_deno)
        # C_model.deno_grounding.update(rand_deno)

        row.update(evaluate(dev_ids, 'dev', D_model, C_model))
        row.update(evaluate(test_ids, 'test', D_model, C_model))
        table.append(row)
        if debug is True:
            break

    row = {
        'path': 'pretrained',
        'deduplicated': 'duplicate' not in str(path),
    }
    D_model.embedding = D_model.pretrained_embed
    C_model.embedding = C_model.pretrained_embed
    row.update(evaluate(dev_ids, 'dev', D_model, C_model))
    row.update(evaluate(test_ids, 'test', D_model, C_model))
    table.append(row)

    with open(out_path, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=table[0].keys(), dialect=csv.excel_tab)
        writer.writeheader()
        writer.writerows(table)


if __name__ == '__main__':
    main()
