import csv
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from models.ideal_grounded import Decomposer, Recomposer
from models.proxy_grounded import ProxyGroundedDecomposer, ProxyGroundedRecomposer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in-dir', action='store', type=Path)
    args = parser.parse_args()
    out_path = args.in_dir / 'summary.tsv'
    luntz_path = args.in_dir / 'luntz.txt'
    device = 'cuda:0'

    if not args.in_dir.exists():
        raise FileNotFoundError("in_dir doesn't exist?")

    checkpoints = sorted(args.in_dir.glob('epoch*.pt'),
                         key=lambda p: int(p.stem.lstrip('epoch')))
    if len(checkpoints) == 0:
        raise FileNotFoundError('No model with path pattern found at in_dir?')

    query_pairs = [
        # CR Bill/Topic
        ('undocumented', 'illegal_aliens'),
        ('estate_tax', 'death_tax'),
        ('capitalism', 'free_market'),
        ('foreign_trade', 'international_trade'),
        ('public_option', 'governmentrun'),
        ('federal_government', 'washington'),

        # # # CR Proxy
        # ('trickledown', 'cut_taxes'),
        # ('voodoo', 'supplyside'),
        # ('tax_expenditures', 'spending_programs'),
        # ('waterboarding', 'interrogation'),
        # ('socialized_medicine', 'singlepayer'),
        # ('political_speech', 'campaign_spending'),
        # ('star_wars', 'strategic_defense_initiative'),
        # ('nuclear_option', 'constitutional_option'),
    ]
    lf = open(luntz_path, 'w')

    table = []
    for in_path in tqdm(checkpoints):
        model = torch.load(in_path, map_location=device)
        model.device = device

        # row = {'path': in_path}
        # row.update(model.tabulate())
        # table.append(row)

        print(in_path, file=lf)
        deno_correct = 0
        cono_correct = 0
        for q1, q2 in query_pairs:
            pre_sim, deno_sim, cono_sim = model.cf_cos_sim(q1, q2)
            deno_delta = deno_sim - pre_sim
            cono_delta = cono_sim - pre_sim
            if deno_delta > 0:
                deno_correct += 1
            if cono_delta < 0:
                cono_correct += 1
            print(f'{pre_sim:.4f}',
                  f'{deno_sim - pre_sim:+.4f}',
                  f'{cono_sim - pre_sim:+.4f}',
                  q1, q2, sep='\t', file=lf)
        print(f'\t{deno_correct}\t{cono_correct}\n', file=lf)
    lf.close()

    print(model.PE_homogeneity)

    columns = table[1].keys()
    with open(out_path, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns, dialect=csv.excel_tab)
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
