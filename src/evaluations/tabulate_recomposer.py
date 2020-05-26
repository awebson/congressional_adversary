import csv
from pathlib import Path
from typing import Set, List, Dict
import torch

from decomposer import Decomposer, DecomposerConfig
from recomposer import Recomposer, RecomposerConfig
from evaluations.helpers import lazy_load_en_masse, PE, WTI
# from evaluations.euphemism import cherry_pairs

# warnings.simplefilter('ignore')
DEVICE = 'cuda:0'
in_dir = Path('../../results/PN/EWS_recomposer/no overcorrect NUB')
checkpoints = lazy_load_en_masse(
    in_dir,
    # patterns=['*/epoch2.pt', '*/epoch4.pt', '*/epoch6.pt', '*/epoch8.pt', '*/epoch10.pt', '*/epoch20.pt', '*/epoch50.pt', '*/epoch74.pt', '*/epoch100.pt'],
    # patterns=['*/epoch10.pt', '*/epoch20.pt', '*/epoch50.pt', '*/epoch80.pt', '*/epoch100.pt'],
    patterns=['epoch*.pt', ],
    device=DEVICE
)
out_path = in_dir / 'summary.tsv'

# rand_path = Path('../../data/ellie/rand_sample.cr.txt')
# with open(rand_path) as file:
#     rand_words = [word.strip() for word in file if word.strip() in PE.word_to_id]
# # print(len(rand_words))
# dev_path = Path('../../data/ellie/partisan_sample_val.cr.txt')
# with open(dev_path) as file:
#     dev_words = [word.strip() for word in file]
# test_path = Path('../../data/ellie/partisan_sample.cr.txt')
# with open(test_path) as file:
#     test_words = [word.strip() for word in file]


rand_path = Path('../../data/ellie/rand_sample.hp.txt')
with open(rand_path) as file:
    rand_words = [word.strip() for word in file]
dev_path = Path('../../data/ellie/partisan_sample_val.hp.txt')
with open(dev_path) as file:
    dev_words = [word.strip() for word in file]
test_path = Path('../../data/ellie/partisan_sample.hp.txt')
with open(test_path) as file:
    test_words = [word.strip() for word in file]


rand_ids = torch.tensor([WTI[w] for w in rand_words], device=DEVICE)
dev_ids = torch.tensor([WTI[w] for w in dev_words], device=DEVICE)
test_ids = torch.tensor([WTI[w] for w in test_words], device=DEVICE)

debug = 0
table: List[Dict] = []
for model in checkpoints:
    D_model = model.deno_decomposer
    C_model = model.cono_decomposer

    if 'pretrained' in model.name:
        row = {'model_name': model.name}

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

        rand_deno = pretrained_neighbors(rand_ids)
        dev_deno = pretrained_neighbors(dev_ids)
        test_deno = pretrained_neighbors(test_ids)

    else:
        row = {
            'model_name': model.name,
            # 'epoch': D_model.stem,
            'DS delta': D_model.delta,
            'DS gamma': D_model.gamma,
            # 'DS delta/gamma': D_model.delta / D_model.gamma,
            'CS delta': C_model.delta,
            'CS gamma': C_model.gamma,
            # 'CS delta/gamma': C_model.delta / C_model.gamma,
            'rho': model.rho
        }

    D_model.deno_grounding.update(rand_deno)
    D_model.deno_grounding.update(dev_deno)
    D_model.deno_grounding.update(test_deno)
    C_model.deno_grounding.update(rand_deno)
    C_model.deno_grounding.update(dev_deno)
    C_model.deno_grounding.update(test_deno)

    row.update(model.tabulate(dev_ids, ' (dev)'))
    row.update(model.tabulate(rand_ids, ' (random)'))
    row.update(model.tabulate(test_ids, ' (test)'))
    table.append(row)
    # if debug > 5:
    #     break
    # debug += 1

columns = table[1].keys()
# columns = list(table[1].keys()) + list(table[0].keys())
with open(out_path, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=columns, dialect=csv.excel_tab)
    writer.writeheader()
    writer.writerows(table)
