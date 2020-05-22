import argparse
from pathlib import Path

from decomposer import Decomposer, DecomposerConfig
from helpers import load_recomposer, PE

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

if args.pretrained:
    from helpers import PE_embed as embed
else:
    D_embed, C_embed = load_recomposer(Path(args.in_file))
    if args.cono:
        embed = C_embed
    else:
        embed = D_embed

with open(args.out_file, 'w') as out_file:
    for word_id, vector in enumerate(embed):
        word = PE.id_to_word[word_id]
        print(word, *vector, sep=' ', file=out_file)
