from pathlib import Path
import torch

from decomposer import Decomposer, DecomposerConfig
from recomposer import Recomposer, RecomposerConfig

in_dir = Path('../results/PN/decomposer/-3c L4')
in_path = in_dir / 'epoch1.pt'
out_path = in_dir / 'denotation_ep1.txt'

model = torch.load(in_path, map_location='cpu')['model']
embed = model.embedding.weight.detach().cpu().numpy()
id_to_word = model.id_to_word

# embed = model.deno_decomposer.embedding.weight.detach().cpu().numpy()
# id_to_word = model.deno_decomposer.id_to_word

with open(out_path, 'w') as out_file:
    for word_id, vector in enumerate(embed):
        word = id_to_word[word_id]
        print(word, *vector, sep=' ', file=out_file)
