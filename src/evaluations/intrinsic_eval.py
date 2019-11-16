import pickle
import csv
import os
from typing import Set, Tuple, NamedTuple, List, Dict, Counter, Optional

import torch
import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr

from decomposer import AdversarialDecomposer, AdversarialConfig

np.random.seed(42)
torch.manual_seed(42)


class Embedding():

    def __init__(self, path: str, source: Optional[str] = None):
        if source is None or source == 'adversarial':
            self.init_from_adversarial(path)
        elif source == 'skip_gram':
            self.init_from_skip_gram(path)
        elif source == 'plain_text':
            self.init_from_plain_text(path)
        else:
            raise ValueError('Unknown embedding source.')

    def init_from_adversarial(self, path: str, device=torch.device('cpu')):
        payload = torch.load(path, map_location=device)
        model = payload['model']
        self.word_to_id = model.word_to_id
        self.id_to_word = model.id_to_word
        self.Dem_frequency: Counter[str] = model.Dem_frequency
        self.GOP_frequency: Counter[str] = model.GOP_frequency

        # encoded layer
        self.embedding = model.export_encoded_embedding(device=device)
#         self.embedding = model.export_decomposed_embedding(device=device)

    def init_from_skip_gram(self, paths: Tuple[str, str]) -> None:
        """Directly extract the weights of a single layer."""
        model_path, vocab_path = paths
        with open(model_path, 'rb') as model_file:
            state_dict = torch.load(model_file, map_location='cpu')
    #     print(state_dict.keys())
        self.embedding = state_dict['center_embedding.weight'].numpy()
        with open(vocab_path, 'rb') as vocab_file:
            self.word_to_id, self.id_to_word, _ = pickle.load(vocab_file)

    def init_from_plain_text(self, path: str) -> Tuple[np.array, Dict[str, int]]:
        id_generator = 0
        word_to_id: Dict[str, int] = {}
        embeddings: List[float] = []
        embedding_file = open(path)
        vocab_size, num_dimensions = map(int, embedding_file.readline().split())
        print(f'vocab_size = {vocab_size:,}, num_dimensions = {num_dimensions}')
        print(f'Loading embeddings from {path}', flush=True)
        for line in embedding_file:
            line: List[str] = line.split()  # type: ignore
            word = line[0]
            vector = np.array(line[-num_dimensions:], dtype=np.float64)
            embeddings.append(vector)
            word_to_id[word] = id_generator
            id_generator += 1
        embedding_file.close()
        print('Done')
        self.id_to_word = {val: key for key, val in word_to_id.items()}
        self.word_to_id = word_to_id
        self.embedding = np.array(embeddings)

    def write_to_tensorboard_projector(self, tb_dir: str) -> None:
        from torch.utils import tensorboard
        tb = tensorboard.SummaryWriter(log_dir=tb_dir)
        all_vocab_ids = range(len(self.word_to_id))
        embedding_labels = [
            self.id_to_word[word_id]
            for word_id in all_vocab_ids]
        tb.add_embedding(
            self.embedding[:9999],
            embedding_labels[:9999],
            global_step=0)

    def cosine_similarity(self, query1: str, query2: str) -> float:
        try:
            query1_id = self.word_to_id[query1]
        except KeyError as error:
            print(f'Out of vocabulary: {query1}')
            raise error
        try:
            query2_id = self.word_to_id[query2]
        except KeyError as error:
            print(f'Out of vocabulary: {query2}')
            raise error
        vectors = self.embedding[(query1_id, query2_id), :]
        similarity = 1 - distance.cosine(vectors[0], vectors[1])
        return similarity

    def nearest_neighbor(self, query: str, top_k: int = 10):
        try:
            query_id = self.word_to_id[query]
        except KeyError:
            raise KeyError(f'{query} is out of vocabulary. Sorry!')
        query_vec = self.embedding[query_id]

        distances = [distance.cosine(query_vec, vec)
                     for vec in self.embedding]
        neighbors = np.argsort(distances)
        print(f"{query}'s neareset neighbors:")
        for ranking in range(1, top_k + 1):
            word_id = neighbors[ranking]
            word = self.id_to_word[word_id]
            cosine_similarity = 1 - distances[word_id]
            print(f'{cosine_similarity:.4f}\t{word}')
        print()


class PhrasePair(NamedTuple):
    query: str
    neighbor: str
    deno_sim: float
    cono_sim: float


def load_cherry(path, exclude_hard_examples=True):
    data = []
    with open(path) as file:
        if path.endswith('tsv'):
            reader = csv.DictReader(file, dialect=csv.excel_tab)
        else:
            reader = csv.DictReader(file)
        for row in reader:
            if row['semantic_similarity'] and row['cono_similarity']:
                if (exclude_hard_examples and
                        'hard example' in row['comment'].lower()):
                    continue
                data.append(PhrasePair(
                    row['query'],
                    row['neighbor'],
#                     row['query_words'],
#                     row['neighbor_words'],
                    float(row['semantic_similarity']),
                    float(row['cono_similarity'])))
    print(f'Loaded {len(data)} labeled entries at {path}')
    return data


def load_MTurk_result(path):
    data = []
    with open(path) as file:
        if path.endswith('tsv'):
            reader = csv.DictReader(file, dialect=csv.excel_tab)
        else:
            reader = csv.DictReader(file)
        for row in reader:
            qc = row['median_query_cono']
            nc = row['median_neighbor_cono']
            if qc and nc:  # nonempty string
                qc = float(qc)
                nc = float(nc)
                if qc == 0 or nc == 0:  # unable to judge
                    continue

                # cono_sim = 5 - abs(qc - nc)

                if ((qc > 3 and nc > 3)
                        or (qc < 3 and nc < 3)
                        or (qc == 3 and nc == 3)):
                    cono_sim = 5
                else:
                    cono_sim = 1

                data.append(PhrasePair(
                    row['query_words'],
                    row['neighbor_words'],
                    float(row['median_deno']),
                    cono_sim))
    print(f'Loaded {len(data)} labeled entries at {path}')
    return data


def correlate_sim_deltas(model, ref_model, phrase_pairs, verbose=False):
    label_deltas = []
    model_deltas = []
    if verbose:
        print(f'deno_sim\tcono_sim\tref_sim\tmodel_sim')

    for pair in phrase_pairs:
        try:
            sim = model.cosine_similarity(pair.query, pair.neighbor)
            ref_sim = ref_model.cosine_similarity(pair.query, pair.neighbor)
        except KeyError:
            continue
        model_delta = sim - ref_sim
        model_deltas.append(model_delta)
        label_deltas.append(pair.deno_sim - pair.cono_sim)

        if verbose:
            print(f'{pair.deno_sim}  {pair.cono_sim}  {ref_sim:.2%}  {sim:.2%}  '
                  f'{pair.query}  {pair.neighbor}')

    median = np.median(model_deltas)
    mean = np.mean(model_deltas)
    stddev = np.std(model_deltas)
    rho, _ = spearmanr(model_deltas, label_deltas)
    return rho, median, mean, stddev


def preview(things):
    for stuff in things:
        q, n, d, c = stuff
        print(d, c, q, n, sep='\t')


def same_deno(pair):
    return pair.deno_sim >= 3


def same_cono(pair):
    return pair.cono_sim >= 3


def is_euphemism(pair) -> bool:
    return same_deno(pair) and not same_cono(pair)


def is_party_platform(pair) -> bool:
    return not same_deno(pair) and same_cono(pair)
