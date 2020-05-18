import random
from pathlib import Path
from typing import Union, List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from helpers import GroundedWord

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
        ax.annotate(w.word, coord, fontsize=8)
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
        ax.annotate(w.word, coord, fontsize=8)
    with open(path, 'wb') as file:
        fig.savefig(file, dpi=300)
    plt.close(fig)


def graph_en_masse(
        models: Dict[str, np.ndarray],
        out_dir: Path,
        reduction: str,  # 'PCA', 'TSNE', or 'both'
        # word_ids: List[int],
        words: List[GroundedWord],
        # hues: Union[List[float], List[int]],
        # sizes: List[int],
        perplexity: Optional[int] = None,
        categorical: bool = False
        ) -> None:
    Path.mkdir(out_dir, parents=True, exist_ok=True)
    word_ids = np.array([w.word_id for w in words])
    for model_name, embed in tqdm(models.items()):
        space = embed[word_ids]
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
        if categorical:
            plot_categorical(visual, words, out_dir / f'{model_name}.png')
        else:
            plot(visual, words, out_dir / f'{model_name}.png')
