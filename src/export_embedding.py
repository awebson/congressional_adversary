from pathlib import Path
import torch

from decomposer import Decomposer, DecomposerConfig
from recomposer import Recomposer, RecomposerConfig

# in_dir = Path('../results/CR_topic/decomposer/Dd4.0 Dg-5.2 Cd-6.8 Cg1.1 R0.0 dp0.26')  # deno sans recomp
in_dir = Path('../results/CR_topic/decomposer/Dd2.7 Dg-40.8 Cd-0.3 Cg7.8 R0.0 dp0.54')  # cono sans recomp
# in_dir = Path('../results/CR_topic/decomposer/Dd4.0 Dg-5.2 Cd-6.8 Cg1.1 R0.0 dp0.26')
in_path = in_dir / 'epoch10.pt'
out_path = in_dir / 'connotation.txt'

model = torch.load(in_path, map_location='cpu')['model']

# Single Decomposer
# embed = model.embedding.weight.detach().cpu().numpy()
# id_to_word = model.id_to_word

# Recomposer
embed = model.cono_decomposer.embedding.weight.detach().cpu().numpy()
id_to_word = model.deno_decomposer.id_to_word

# Pretrained
# embed = model.pretrained_embed.weight.detach().cpu().numpy()

with open(out_path, 'w') as out_file:
    for word_id, vector in enumerate(embed):
        word = id_to_word[word_id]
        print(word, *vector, sep=' ', file=out_file)
