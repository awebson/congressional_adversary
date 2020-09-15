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
from models.ideal_grounded import Decomposer, Recomposer

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
        fig.savefig(file, dpi=300)
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
        fig.savefig(file, dpi=300)
    plt.close(fig)


def graph_en_masse(
        spaces: Dict[str, torch.tensor],
        out_dir: Path,
        reduction: str,  # 'PCA', 'TSNE', or 'both'
        # word_ids: List[int],
        words: List[GroundedWord],
        # hues: Union[List[float], List[int]],
        # sizes: List[int],
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
                perplexity=perplexity, learning_rate=10,
                n_iter=5000, n_iter_without_progress=1000).fit_transform(space)
        elif reduction == 'both':
            assert perplexity is not None
            space = PCA(n_components=30).fit_transform(space)
            visual = TSNE(
                perplexity=perplexity, learning_rate=10,
                n_iter=5000, n_iter_without_progress=1000).fit_transform(space)
        else:
            raise ValueError('unknown dimension reduction method')
        if color_code == 'deno':
            plot_categorical(visual, words, out_dir / f'{name}.png')
        elif color_code == 'cono':
            plot(visual, words, out_dir / f'{name}.png')
        else:
            raise ValueError


def main():
    in_path = Path('../../results/replica/CR_topic/new ground/epoch100.pt')
    out_dir = in_path.parent
    model = torch.load(in_path, map_location='cpu')

    query_words = [
        'government', 'washington',
        'estate_tax', 'death_tax',
        'public_option', 'governmentrun',
        'foreign_trade', 'international_trade',
        # 'cut_taxes', 'supply_side' #'trickledown'
        'undocumented', 'illegal_aliens',
        'capitalism', 'free_market'
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

    graph_en_masse(
        spaces, out_dir=out_dir / 'topic/t-SNE p3', color_code='deno',
        reduction='TSNE', perplexity=3, words=grounded_words)

    graph_en_masse(
        spaces, out_dir=out_dir / 'party/t-SNE p3', color_code='cono',
        reduction='TSNE', perplexity=3, words=grounded_words)

if __name__ == "__main__":
    main()
