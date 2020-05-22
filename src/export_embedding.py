import argparse
from pathlib import Path

from decomposer import Decomposer, DecomposerConfig
from recomposer import Recomposer, RecomposerConfig
# from helpers import load, load_recomposer, PE

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--in-file', action='store', type=str)
parser.add_argument(
    '-o', '--out-file', action='store', type=str)
parser.add_argument(
    '-p', '--pretrained', action='store_true')
parser.add_argument(
    '-c', '--cono', action='store_true',
    help='Export connotation vectors instead of denotation vectors.')
args = parser.parse_args()

# if args.pretrained:
#     from helpers import PE_embed as embed
# else:
#     D_embed, C_embed = load_recomposer(Path(args.in_file))
#     if args.cono:
#         embed = C_embed
#     else:
#         embed = D_embed
# id_to_word = PE.id_to_word

# args.in_file = '../results/news/validation/-3c BS1024/epoch15.pt'
# args.out_file = '../results/exported_embeddings/denotation_e15.txt'
# args.in_file = '../results/3bins recomp/clamp L4/epoch7.pt'
# args.out_file = '../results/exported_embeddings/denotation_RC_L4_e7.txt'

# args.in_file = '../results/congress bill topic/defense/Dd1.9 Dg-2.5 Cd-2.7 Cg6.0 R16.2 dp0.09/epoch10.pt'
# args.out_file = '../results/exported_embeddings/for_real/connotation.txt'
# args.in_file = '../results/search/Dd1.4 Dg-0.8 Cd-0.1 Cg9.7 R69.6 MAL6.9/epoch2.pt'
# args.out_file = '../results/exported_embeddings/for_real/PN_denotation.txt'

# args.in_file = '../results/PN/EWS_recomposer/L4/epoch1.pt'
# args.out_file = '../results/exported_embeddings/PN_EWS_L4_denotation.txt'

args.in_file = '../results/PN/decomposer/-3c L1/epoch1.pt'
args.out_file = '../results/exported_embeddings/PN_0R_L1_denotation.txt'


import torch
model = torch.load(Path(args.in_file), map_location='cpu')['model']
embed = model.embedding.weight.detach().cpu().numpy()
id_to_word = model.id_to_word
# embed = model.deno_decomposer.embedding.weight.detach().cpu().numpy()
# id_to_word = model.deno_decomposer.id_to_word

with open(args.out_file, 'w') as out_file:
    for word_id, vector in enumerate(embed):
        word = id_to_word[word_id]
        print(word, *vector, sep=' ', file=out_file)
