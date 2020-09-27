import argparse
from pathlib import Path

import torch

from data import GroundedWord
from models.ideal_grounded import Decomposer, Recomposer
from models.proxy_grounded import ProxyGroundedDecomposer, ProxyGroundedRecomposer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--in-path', action='store', type=Path)
    args = parser.parse_args()
    in_path = args.in_path

    # in_path = Path('../../results/replica/CR_topic/new code torch1.5/epoch100.pt')
    # in_path = Path('../../results/replica/CR_topic/ctx5/epoch100.pt')
    # in_path = Path('../../results/replica/CR_topic/LinRe/epoch100.pt')
    # in_path = Path('../../results/replica/CR_topic/ctx5 init old HS/epoch140.pt')
    # in_path = Path('../../results/replica/CR_topic/ctx0 init ctx3 HS/epoch150.pt')
    # in_path = Path('../../results/camera/CR_topic/ctx3 LinRe LR1e-5 c5 init/epoch5.pt')

    # in_path = Path('../../results/replica/CR_bill/ctx0 init SGNS/epoch150.pt')
    # in_path = Path('../../results/replica/CR_bill/ctx3 L2 replica super SGNS/epoch150.pt')

    # in_path = Path('../../results/replica/CR_skip/clip none extra B8k/epoch6.pt')
    # in_path = Path('../../results/replica/CR_skip/naive LM B512 LR1e-3/epoch2_4.pt')
    # in_path = Path('../../results/replica/CR_skip/LinRe/epoch26.pt')  # best?
    # in_path = Path('../../results/replica/CR_skip/ctx embed/epoch5.pt')
    # in_path = Path('../../results/replica/CR_skip/PE SGNS 97/epoch2.pt')
    # in_path = Path('../../results/replica/CR_skip/naive LM/epoch2_12.pt')
    # in_path = Path('../../results/replica/CR_skip/naive LM B512 LR1e-3/epoch1.pt')

    # in_path = Path('../../results/replica/PN_skip/clip all/epoch20.pt')
    out_dir = in_path.parent
    model = torch.load(in_path, map_location='cpu')
    print(in_path)

    # call tabluate

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

    # import IPython
    # IPython.embed()

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
