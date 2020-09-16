import random
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from data import GroundedWord
# from models.ideal_grounded import Decomposer, Recomposer
from models.proxy_grounded import ProxyGroundedDecomposer, ProxyGroundedRecomposer

random.seed(42)
np.random.seed(42)
sns.set()


def plot(
        coordinates: np.ndarray,
        words: List[GroundedWord],
        path: Path
        ) -> None:
    fig, ax = plt.subplots(figsize=(15, 10))
    skew = [w.R_ratio for w in words]
    freq = [w.freq for w in words]
    sns.scatterplot(
        coordinates[:, 0], coordinates[:, 1],
        hue=skew, palette='coolwarm',  # hue_norm=(0, 1),
        size=freq, sizes=(200, 1000),
        legend=None, ax=ax)
    for coord, w in zip(coordinates, words):
        ax.annotate(w.text, coord, fontsize=8)
    with open(path, 'wb') as file:
        fig.savefig(file, dpi=200)
    plt.close(fig)


def plot_categorical(
        coordinates: np.ndarray,
        words: List[GroundedWord],
        path: Path
        ) -> None:
    fig, ax = plt.subplots(figsize=(20, 10))
    categories = [w.majority_deno for w in words]
    freq = [w.freq for w in words]
    sns.scatterplot(
        coordinates[:, 0], coordinates[:, 1],
        hue=categories, palette='muted', hue_norm=(0, 1),
        size=freq, sizes=(200, 1000),
        legend='brief', ax=ax)
    chartBox = ax.get_position()
    ax.set_position(  # adjust legend
        [chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), ncol=1)
    for coord, w in zip(coordinates, words):
        ax.annotate(w.text, coord, fontsize=8)
    with open(path, 'wb') as file:
        fig.savefig(file, dpi=200)
    plt.close(fig)


def graph_en_masse(
        spaces: Dict[str, torch.tensor],
        out_dir: Path,
        reduction: str,  # 'PCA', 'TSNE', or 'both'
        words: List[GroundedWord],
        color_code: str,
        perplexity: Optional[int] = None,
        ) -> None:
    Path.mkdir(out_dir, parents=True, exist_ok=True)

    for name, space in tqdm(spaces.items()):
        space = space.detach().numpy()
        if reduction == 'PCA':
            visual = PCA(n_components=2).fit_transform(space)
        elif reduction == 'TSNE':
            assert perplexity is not None
            visual = TSNE(
                perplexity=perplexity, learning_rate=1,
                n_iter=5000, n_iter_without_progress=1000).fit_transform(space)
        elif reduction == 'both':
            assert perplexity is not None
            space = PCA(n_components=30).fit_transform(space)
            visual = TSNE(
                perplexity=perplexity, learning_rate=1,
                n_iter=5000, n_iter_without_progress=1000).fit_transform(space)
        else:
            raise ValueError('unknown dimension reduction method')

        if color_code == 'both':
            plot_categorical(visual, words, out_dir / f'topic {name}.png')
            plot(visual, words, out_dir / f'party {name}.png')
        elif color_code == 'deno':
            plot_categorical(visual, words, out_dir / f'{name}.png')
        elif color_code == 'cono':
            plot(visual, words, out_dir / f'{name}.png')
        else:
            raise ValueError


def main():
    # in_path = Path('../../results/replica/CR_topic/ctx5/epoch100.pt')
    # in_path = Path('../../results/replica/CR_bill/new ground/epoch50.pt')
    # in_path = Path('../../results/replica/CR_skip/clip none extra/epoch6.pt')
    # in_path = Path('../../results/replica/CR_skip/clip none extra B8k/epoch12.pt')
    in_path = Path('../../results/replica/CR_skip/PE SGNS 97/epoch2.pt')
    # in_path = Path('../../results/replica/PN_skip/clip all/epoch20.pt')
    out_dir = in_path.parent
    model = torch.load(in_path, map_location='cpu')
    # model.word_to_id = {v: k for k, v in model.id_to_word.items()}

    query_words = [
        # CR
        'federal_government', 'washington',
        'estate_tax', 'death_tax',
        'public_option', 'governmentrun',
        'foreign_trade', 'international_trade',
        'undocumented', 'illegal_aliens',
        'capitalism', 'free_market',

        'trickledown', 'cut_taxes',
        'voodoo', 'supplyside',
        'tax_expenditures', 'spending_programs',
        'waterboarding', 'interrogation',
        'socialized_medicine', 'singlepayer',
        'political_speech', 'campaign_spending',
        'star_wars', 'strategic_defense_initiative',
        'nuclear_option', 'constitutional_option'

        # # PN
        # 'government', 'washington',
        # 'estate_tax', 'death_tax',
        # 'public_option', 'government_run',
        # 'foreign_trade', 'international_trade',
        # 'undocumented_workers', 'illegal_aliens',
        # 'capitalism', 'free_market',

        # 'cut_taxes', 'voodoo', 'supply_side',  # 'trickledown',
        # 'tax_expenditures', 'spending_programs',
        # 'waterboarding', 'interrogation',
        # 'socialized_medicine', 'single_payer',
        # 'political_speech', 'campaign_spending',

    ]
    query_ids = torch.tensor([model.word_to_id[w] for w in query_words])
    spaces = {
        'pretrained_space': model.pretrained_embed(query_ids),
        'deno_space': model.deno_space.decomposed(query_ids),
        'cono_space': model.cono_space.decomposed(query_ids)
    }

    grounded_words = [model.ground[w] for w in query_words]
    for w in grounded_words:
        w.init_plotting()
        # print(w)

    graph_en_masse(
        spaces, out_dir=out_dir / 'PCA', color_code='both',
        reduction='PCA', words=grounded_words)

    for perplexity in (2, 4, 5, 6, 8, 10, 12):
        graph_en_masse(
            spaces, out_dir=out_dir / f't-SNE p{perplexity}', color_code='cono',
            reduction='TSNE', perplexity=perplexity, words=grounded_words)

if __name__ == "__main__":
    main()
