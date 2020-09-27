import argparse
from pathlib import Path

import torch

from models.ideal_grounded import Decomposer, Recomposer
from models.proxy_grounded import ProxyGroundedDecomposer, ProxyGroundedRecomposer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in-path', action='store', type=Path)
    args = parser.parse_args()

    # in_path = Path('../../results/replica/CR_skip/LinRe/epoch26.pt')
    model = torch.load(args.in_path, map_location='cpu')
    print(args.in_path)

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
              f'{deno_sim:.4f}',
              f'{cono_sim:.4f}',
              q1, q2, sep='\t')
    print(f'\t{deno_correct}\t{cono_correct}')


if __name__ == "__main__":
    main()
