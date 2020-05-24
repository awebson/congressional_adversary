import csv
from pathlib import Path
from typing import Set, Tuple, List, Dict

import numpy as np
import torch
from tqdm import tqdm

# from decomposer import Decomposer, DecomposerConfig
# in_dir = Path('../../results/PN/decomposer')
# patterns = ['*/epoch*.pt']
# out_path = in_dir / 'summary.tsv'
# random_path = Path('../../data/ellie/rand_sample.hp.txt')
# dev_path = Path('../../data/ellie/partisan_sample_val.hp.txt')
# test_path = Path('../../data/ellie/partisan_sample.hp.txt')


from old_congress import Decomposer, DecomposerConfig
device = torch.device('cuda:0')
in_dir = Path('../../results/CR_skip/sans recomposer')
patterns = ['*/epoch*.pt']
out_path = in_dir / 'summary.tsv'
random_path = Path('../../data/ellie/rand_sample.cr.txt')
dev_path = Path('../../data/ellie/partisan_sample_val.cr.txt')
test_path = Path('../../data/ellie/partisan_sample.cr.txt')


# def main() -> None:
checkpoints: List[Path] = []
for pattern in patterns:
    checkpoints += list(in_dir.glob(pattern))
if len(checkpoints) == 0:
    raise FileNotFoundError(f'No model with path pattern found at {in_dir}?')

with open(random_path) as file:
    random_words = [word.strip() for word in file]
with open(dev_path) as file:
    dev_words = [word.strip() for word in file]
with open(test_path) as file:
    test_words = [word.strip() for word in file]

debug = 0
wid = None
table: List[Dict] = []
for path in tqdm(checkpoints):
    tqdm.write(f'Loading {path}')
    payload = torch.load(path, map_location=device)
    model = payload['model']
    config = payload['config']
    row = {
        'path': path.parent.name + '/' + path.name,  # path.parent.name
        'delta': config.delta,
        'gamma': config.gamma,
        'max_adversary_loss': config.max_adversary_loss,
        'batch size': config.batch_size,
        'learning rate': config.learning_rate,
    }

    if wid is not None:
        assert wid == model.word_to_id
    else:
        wid = model.word_to_id
        rand_ids = torch.tensor([wid[word] for word in random_words], device=device)
        dev_ids = torch.tensor([wid[word] for word in dev_words], device=device)
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
                qv = model.pretrained_embed(qid)
                qid = qid.item()
                qw = model.id_to_word[qid]
                cos_sim = F.cosine_similarity(qv.unsqueeze(0), model.pretrained_embed.weight)
                cos_sim, neighbor_ids = cos_sim.topk(k=top_k + 5, dim=-1)
                neighbor_ids = [
                    nid for nid in neighbor_ids.tolist()
                    if editdistance.eval(qw, model.id_to_word[nid]) > 3]
                deno_grounding[qid] = set(neighbor_ids[:top_k])
            return deno_grounding

        rand_deno = pretrained_neighbors(rand_ids)
        dev_deno = pretrained_neighbors(dev_ids)
        test_deno = pretrained_neighbors(test_ids)
    # End checking vocabulary equality

    model.deno_grounding = rand_deno  # NOTE
    model.deno_grounding.update(dev_deno)
    model.deno_grounding.update(test_deno)

    row.update(model.tabulate(dev_ids, ' (dev)'))
    row.update(model.tabulate(rand_ids, ' (random)'))
    row.update(model.tabulate(test_ids, ' (test)'))

    table.append(row)
    # if debug > 5:
    #     break
    # debug += 1

row = {'path': 'pretrained'}
model.embedding = model.pretrained_embed
row.update(model.tabulate(dev_ids, ' (dev)'))
row.update(model.tabulate(rand_ids, ' (random)'))
row.update(model.tabulate(test_ids, ' (test)'))
table.append(row)

with open(out_path, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=table[0].keys(), dialect=csv.excel_tab)
    writer.writeheader()
    writer.writerows(table)


# if __name__ == '__main__':
#     main()
