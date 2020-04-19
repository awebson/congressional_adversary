import argparse
from pathlib import Path

from decomposer import Decomposer, DecomposerConfig
from recomposer import Recomposer, RecomposerConfig
from helpers import load, load_recomposer, PE

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

import torch
model = torch.load(Path(args.in_file), map_location='cpu')['model']
embed = model.deno_decomposer.embedding.weight.detach().cpu().numpy()
id_to_word = model.id_to_word

with open(args.out_file, 'w') as out_file:
    for word_id, vector in enumerate(embed):
        word = id_to_word[word_id]
        print(word, *vector, sep=' ', file=out_file)
