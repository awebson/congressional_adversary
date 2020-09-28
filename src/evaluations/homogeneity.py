import csv
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from models.ideal_grounded import Decomposer, Recomposer
from models.proxy_grounded import ProxyGroundedDecomposer, ProxyGroundedRecomposer

query_pairs = [
    # CR Bill/Topic
    ('undocumented', 'illegal_aliens'),
    ('estate_tax', 'death_tax'),
    ('capitalism', 'free_market'),
    ('foreign_trade', 'international_trade'),
    ('public_option', 'governmentrun'),
    ('federal_government', 'washington'),

    # CR Proxy
    ('trickledown', 'cut_taxes'),
    ('voodoo', 'supplyside'),
    ('tax_expenditures', 'spending_programs'),
    ('waterboarding', 'interrogation'),
    ('socialized_medicine', 'singlepayer'),
    ('political_speech', 'campaign_spending'),
    ('star_wars', 'strategic_defense_initiative'),
    ('nuclear_option', 'constitutional_option'),

    # # PN Proxy
    # ('undocumented_workers', 'illegal_aliens'),
    # ('estate_tax', 'death_tax'),
    # ('capitalism', 'free_market'),
    # ('foreign_trade', 'international_trade'),
    # ('public_option', 'government_run'),
    # ('federal_government', 'washington'),
    # ('supply_side', 'cut_taxes'),
    # ('voodoo', 'supply_side'),
    # ('tax_expenditures', 'spending_programs'),
    # ('waterboarding', 'interrogation'),
    # ('socialized_medicine', 'single_payer'),
    # ('political_speech', 'campaign_spending')
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in-dir', action='store', type=Path)
    parser.add_argument('-f', '--fast', action='store_true')
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

    lf = open(luntz_path, 'w')

    table = []
    for in_path in tqdm(checkpoints):
        model = torch.load(in_path, map_location=device)
        model.device = device

        if not args.fast:
            row = {'path': in_path}
            row.update(model.tabulate())
            table.append(row)

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

    if args.fast:
        return

    columns = table[1].keys()
    with open(out_path, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns, dialect=csv.excel_tab)
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
