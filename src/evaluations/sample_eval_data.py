from typing import Tuple, List, Dict

import numpy as np
from sklearn.metrics import pairwise

np.random.seed(42)


class Phrase:

    def __init__(self, line: str, min_freq: int):
        data = line.strip().split('\t')
        self.D_ratio = float(data[0])
        self.R_ratio = float(data[1])
        self.D_freq = int(data[2])
        self.R_freq = int(data[3])
        total = self.D_freq + self.R_freq
        if total < min_freq:
            raise StopIteration
        self.words = data[4]
        try:
            self.skew = self.R_freq / self.D_freq
        except ZeroDivisionError:
            self.skew = self.R_freq / 1e-5


class Embedding():

    def __init__(self, embed_path: str, party_freq_path: str, min_freq: int):
        id_generator = 0
        word_to_id: Dict[str, int] = {}
        embedding: List[np.array] = []
        with open(embed_path) as embedding_file:
            vocab_size, num_dimensions = map(int, embedding_file.readline().split())
            print(f'Loading pretrained embeddings from {embed_path}')
            print(f'vocab_size = {vocab_size:,}, num_dimensions = {num_dimensions}')
            for line in embedding_file:
                line: List[str] = line.split()
                word = line[0]
                vector = np.array(line[-num_dimensions:], dtype=np.float64)
                embedding.append(vector)
                word_to_id[word] = id_generator
                id_generator += 1
        self.id_to_word = {val: key for key, val in word_to_id.items()}
        self.word_to_id = word_to_id
        self.embedding = np.array(embedding)

        # Load partisan frequncy file as grounded partisanship
        self.phrase_to_id: Dict[Phrase, int] = {}
        self.id_to_phrase: Dict[int, Phrase] = {}
        with open(party_freq_path) as vocab_file:
            vocab_file.readline()  # skip header line
            for line in vocab_file:
                try:
                    phrase = Phrase(line, min_freq)
                except StopIteration:
                    continue
                phrase_id = self.word_to_id[phrase.words]
                self.id_to_phrase[phrase_id] = phrase
        print(f'with min_frequency = {min_freq}, '
              f'frequency file vocab size = {len(self.id_to_phrase)}')

    def export_nearest_partisan_neighbors(
            self,
            export_path: str,
            query_ids: List[int],
            target_ids: List[int],
            top_k: int = 10
            ) -> None:
        query_embed = self.embedding[query_ids]
        target_embed = self.embedding[target_ids]
        cosine_sim = pairwise.cosine_similarity(query_embed, target_embed)
        top_neighbor_ids = np.argsort(-cosine_sim)  # negate to reverse sort order
        top_neighbor_ids = top_neighbor_ids[:, 1:top_k + 1]  # excludes matrix diagonal (distance to self)

        output: Dict[int, List[Tuple[float, int]]] = {}
        for query_index, sorted_target_indices in enumerate(top_neighbor_ids):
            query_id = query_ids[query_index]
            output[query_id] = []
            for sort_rank, target_index in enumerate(sorted_target_indices):
                target_id = target_ids[target_index]
                output[query_id].append(
                    (target_id, cosine_sim[query_index][target_index]))

        file = open(export_path, 'w')
        file.write(
            'same_connotation\tsame_denotation\t'  # blank columns for labeling
            'query\tcosine_similarity\tneighbor\t'
            'query_D_freq\tquery_R_freq\tquery_R_ratio\t'
            'neighbor_D_freq\tneighbor_R_freq\tneighbor_R_ratio\n')
        for query_id, neighbor_ids in output.items():
            query = self.id_to_phrase[query_id]
            for neighbor_id, cosine_sim in neighbor_ids:
                neighbor = self.id_to_phrase[neighbor_id]
                file.write(
                    f'\t\t{query.words}\t{cosine_sim:.4}\t{neighbor.words}\t'
                    f'{query.D_freq}\t{query.R_freq}\t{query.R_ratio}\t'
                    f'{neighbor.D_freq}\t{neighbor.R_freq}\t{neighbor.R_ratio}\n')
        file.close()

    def export_combined_nearest_partisan_neighbors(
            self,
            export_path: str,
            query_ids: List[int],
            Dem_ids: List[int],
            GOP_ids: List[int],
            neutral_ids: List[int],
            top_k: int = 10
            ) -> None:

        def helper(target_ids: List[int]) -> Dict[int, List[Tuple[int, float]]]:
            query_embed = self.embedding[query_ids]
            target_embed = self.embedding[target_ids]
            cosine_sim = pairwise.cosine_similarity(query_embed, target_embed)
            top_neighbor_ids = np.argsort(-cosine_sim)  # negate to reverse sort
            # excludes matrix diagonal (distance to self)
            top_neighbor_ids = top_neighbor_ids[:, 1:top_k + 1]

            output: Dict[int, List[Tuple[int, float]]] = {}
            for query_index, sorted_target_indices in enumerate(top_neighbor_ids):
                query_id = query_ids[query_index]
                output[query_id] = []
                for sort_rank, target_index in enumerate(sorted_target_indices):
                    target_id = target_ids[target_index]
                    output[query_id].append(
                        (target_id, cosine_sim[query_index][target_index]))
            return output

        combined_output: Dict[int, List[Tuple[int, float]]] = {}
        for target_ids in (Dem_ids, GOP_ids, neutral_ids):
            for neighbor_id, neighborhood in helper(target_ids).items():
                if neighbor_id not in combined_output:
                    combined_output[neighbor_id] = neighborhood
                else:
                    combined_output[neighbor_id] += neighborhood

        file = open(export_path, 'w')
        file.write(
            'same_connotation\tsame_denotation\t'  # blank columns for labeling
            'query_D_freq\tquery_R_freq\tquery_R_ratio\t'
            'query\tcosine_similarity\tneighbor\t'
            'neighbor_D_freq\tneighbor_R_freq\tneighbor_R_ratio\n')
        for query_id, neighborhood in combined_output.items():
            query = self.id_to_phrase[query_id]
            for neighbor_id, cosine_sim in neighborhood:
                neighbor = self.id_to_phrase[neighbor_id]
                file.write(
                    f'\t\t{query.D_freq}\t{query.R_freq}\t{query.R_ratio:.2%}\t'
                    f'{query.words}\t{cosine_sim:.4}\t{neighbor.words}\t'
                    f'{neighbor.D_freq}\t{neighbor.R_freq}\t{neighbor.R_ratio:.2%}\n')
        file.close()


def main() -> None:
    model = Embedding(
        embed_path='../../data/processed/pretrained_word2vec/for_real.txt',
        party_freq_path='../../data/processed/plain_text/for_real/vocab_partisan_frequency.tsv',
        min_freq=100)

    Dem_ids = []
    GOP_ids = []
    neutral_ids = []
    for phrase_id, phrase in model.id_to_phrase.items():
        if phrase.D_ratio > 0.8:
            Dem_ids.append(phrase_id)
        elif phrase.R_ratio > 0.8:
            GOP_ids.append(phrase_id)
        elif 0.4 < phrase.R_ratio < 0.6:
            neutral_ids.append(phrase_id)

    Dem_ids, GOP_ids, neutral_ids = map(np.array, (Dem_ids, GOP_ids, neutral_ids))
    print(f'{len(GOP_ids)} capitalists\n'
          f'{len(Dem_ids)} socialists\n'
          f'{len(neutral_ids)} neoliberal shills')

    samples_per_party = 200
    sampled_Dem_ids = Dem_ids[np.random.randint(0, len(Dem_ids), samples_per_party)]
    sampled_GOP_ids = GOP_ids[np.random.randint(0, len(GOP_ids), samples_per_party)]
    sampled_neutral_ids = neutral_ids[np.random.randint(0, len(neutral_ids), samples_per_party)]

    base_dir = '../../data/evaluation/'
    model.export_combined_nearest_partisan_neighbors(
        base_dir + 'Dem_sample.tsv', sampled_Dem_ids, Dem_ids, GOP_ids, neutral_ids)
    model.export_combined_nearest_partisan_neighbors(
        base_dir + 'GOP_sample.tsv', sampled_GOP_ids, Dem_ids, GOP_ids, neutral_ids)
    model.export_combined_nearest_partisan_neighbors(
        base_dir + 'neutral_sample.tsv', sampled_neutral_ids, Dem_ids, GOP_ids, neutral_ids)


if __name__ == '__main__':
    main()
