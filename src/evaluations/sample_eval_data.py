import re
import csv
import random
from typing import Tuple, NamedTuple, List, Dict, Optional, Any

import numpy as np
import editdistance
from sklearn.metrics import pairwise
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
remove_underscores = str.maketrans('_', ' ')

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
        # try:
        #     self.skew = self.R_freq / self.D_freq
        # except ZeroDivisionError:
        #     self.skew = self.R_freq / 1e-5


class PhrasePair(NamedTuple):

    query: Phrase
    neighbor: Phrase
    cosine_sim: float

    def export_dict(self) -> Dict:
        data = {
            'query_' + key: val
            for key, val in vars(self.query).items()}
        data.update(
            {'neighbor_' + key: val
             for key, val in vars(self.neighbor).items()})
        data['cosine_sim'] = self.cosine_sim
        return data


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

    def nearest_partisan_neighbors(
            self,
            query_ids: List[int],
            target_ids: List[int],
            top_k: int
            ) -> Dict[int, List[Tuple[int, float]]]:
        query_embed = self.embedding[query_ids]
        target_embed = self.embedding[target_ids]
        cosine_sim = pairwise.cosine_similarity(query_embed, target_embed)
        top_neighbor_ids = np.argsort(-cosine_sim)  # negate to reverse sort
        # top_neighbor_ids = top_neighbor_ids[:, 0:top_k]

        output: Dict[int, List[Tuple[int, float]]] = {}
        for query_index, sorted_target_indices in enumerate(top_neighbor_ids):
            query_id = query_ids[query_index]
            query_words = self.id_to_word[query_id]
            output[query_id] = []
            num_neighbors = 0
            for sort_rank, target_index in enumerate(sorted_target_indices):
                if num_neighbors == top_k:
                    break
                target_id = target_ids[target_index]
                target_words = self.id_to_word[target_id]
                if editdistance.eval(query_words, target_words) < 3:
                    # print(query_words, target_words)
                    continue
                num_neighbors += 1
                output[query_id].append(
                    (target_id, cosine_sim[query_index][target_index]))
        return output

    def export_combined_nearest_partisan_neighbors(
            self,
            query_ids: List[int],
            list_neighbor_ids: List[List[int]],
            top_k: int,
            export_path: Optional[str]
            ) -> Optional[List[PhrasePair]]:
        """Writes CSV to export_path. If export_path is None, returns Dict."""

        combined_output: Dict[int, List[Tuple[int, float]]] = {}
        for target_ids in list_neighbor_ids:
            dist_dict = self.nearest_partisan_neighbors(query_ids, target_ids, top_k)
            for neighbor_id, neighborhood in dist_dict.items():
                if neighbor_id not in combined_output:
                    combined_output[neighbor_id] = neighborhood
                else:
                    combined_output[neighbor_id] += neighborhood

        def sum_freq(item: Tuple[int, Any]) -> int:
            query_id = item[0]
            query = self.id_to_phrase[query_id]
            return query.D_freq + query.R_freq

        if not export_path:
            phrase_pairs: List[PhrasePair] = []
            for query_id, neighborhood in sorted(
                    combined_output.items(), key=sum_freq, reverse=True):
                query = self.id_to_phrase[query_id]
                neighborhood.sort(key=lambda tup: tup[1], reverse=True)
                for neighbor_id, cosine_sim in neighborhood:
                    neighbor = self.id_to_phrase[neighbor_id]
                    if neighbor == query:
                        continue
                    phrase_pairs.append(PhrasePair(query, neighbor, cosine_sim))
            return phrase_pairs

        else:
            file = open(export_path, 'w')
            file.write(
                'comment\tdeno_similarity\tcono_similarity\t'  # blank columns for labeling
                'query_D_freq\tquery_R_freq\tquery_R_ratio\t'
                'query\tcosine_similarity\tneighbor\t'
                'neighbor_D_freq\tneighbor_R_freq\tneighbor_R_ratio\n')

            for query_id, neighborhood in sorted(
                    combined_output.items(), key=sum_freq, reverse=True):
                query = self.id_to_phrase[query_id]
                neighborhood.sort(key=lambda tup: tup[1], reverse=True)
                for neighbor_id, cosine_sim in neighborhood:
                    neighbor = self.id_to_phrase[neighbor_id]
                    if neighbor == query:
                        continue
                    file.write(
                        f'\t\t\t{query.D_freq}\t{query.R_freq}\t{query.R_ratio:.2%}\t'
                        f'{query.words}\t{cosine_sim:.4}\t{neighbor.words}\t'
                        f'{neighbor.D_freq}\t{neighbor.R_freq}\t{neighbor.R_ratio:.2%}\n')
            file.close()
            return None

    def export_MTurk(
            self,
            out_path: str,
            phrase_pairs: List[PhrasePair],
            shuffle: bool
            ) -> None:
        column_names = [
            'query_D_freq', 'query_R_freq', 'query_D_ratio', 'query_R_ratio',
            'query_words', 'cosine_sim', 'neighbor_words',
            'neighbor_D_freq', 'neighbor_R_freq', 'neighbor_D_ratio', 'neighbor_R_ratio'
        ]
        out_file = open(out_path, 'w')
        csv_writer = csv.DictWriter(out_file, column_names)
        csv_writer.writeheader()
        if shuffle:
            random.shuffle(phrase_pairs)
        for pair in tqdm(phrase_pairs):
            csv_writer.writerow(pair.export_dict())
        out_file.close()

    def export_MTurk_with_context_sentences(
            self,
            corpus_path: str,
            out_path: str,
            phrase_pairs: List[PhrasePair],
            contexts_per_phrase: int
            ) -> None:
        with open(corpus_path) as file:
            corpus = [line for line in file]
        random.shuffle(corpus)

        column_names = (
            ['query_D_freq', 'query_R_freq', 'query_D_ratio', 'query_R_ratio',
             'query_words', 'cosine_sim', 'neighbor_words',
             'neighbor_D_freq', 'neighbor_R_freq', 'neighbor_D_ratio', 'neighbor_R_ratio']
            + [f'query_context{i}' for i in range(contexts_per_phrase)]
            + [f'neighbor_context{i}' for i in range(contexts_per_phrase)]
        )
        out_file = open(out_path, 'w')
        csv_writer = csv.DictWriter(out_file, column_names)
        csv_writer.writeheader()

        def code_injection(regex_match) -> str:
            return '<b>' + regex_match.group() + '</b>'

        for pair in tqdm(phrase_pairs, desc='RegExing contexts in corpus'):
            num_contexts = 0
            phrase_pattern = re.compile(pair.query.words)
            context_pattern = re.compile(r'(\w*\s){1,30}' + pair.query.words + r'(\s\w*){1,30}')
            for speech in corpus:
                if pair.query.words not in speech:
                    continue
                context = context_pattern.search(speech)
                if context:
                    context = context.group().strip()
                    context = phrase_pattern.sub(code_injection, context)
                    # context = context.translate(remove_underscores)
                    setattr(pair.query, f'context{num_contexts}', context)
                    num_contexts += 1
                if num_contexts == contexts_per_phrase:
                    break

            num_contexts = 0
            phrase_pattern = re.compile(pair.neighbor.words)
            context_pattern = re.compile(r'(\w*\s){1,30}' + pair.neighbor.words + r'(\s\w*){1,30}')
            for speech in corpus:
                if pair.neighbor.words not in speech:
                    continue
                context = context_pattern.search(speech)
                if context:
                    context = context.group().strip()
                    context = phrase_pattern.sub(code_injection, context)
                    # context = context.translate(remove_underscores)
                    setattr(pair.neighbor, f'context{num_contexts}', context)
                    num_contexts += 1
                if num_contexts == contexts_per_phrase:
                    # pair.neighbor.words = pair.neighbor.words.translate(remove_underscores)
                    break

            # pair.query.words = pair.query.words.translate(remove_underscores)
            csv_writer.writerow(pair.export_dict())
        out_file.close()

    def motivation_analysis(
            self,
            Dem_ids: List[int],
            GOP_ids: List[int],
            threshold: float,
            top_k: int
            ) -> None:
        all_vocab_ids = np.array(list(self.id_to_phrase.keys()))

        Dem_neighborhood = self.nearest_partisan_neighbors(
            Dem_ids, all_vocab_ids, top_k=top_k)
        GOP_neighborood = self.nearest_partisan_neighbors(
            GOP_ids, all_vocab_ids, top_k=top_k)

        D_neighbor_ratios = []
        for query_id, neighborhood in Dem_neighborhood.items():
            query = self.id_to_phrase[query_id]
            cosponsor = 0
            for neighbor_id, cosine_sim in neighborhood:
                neighbor = self.id_to_phrase[neighbor_id]
                if neighbor == query:
                    continue
                if neighbor.D_ratio > threshold and query.D_ratio > threshold:
                    cosponsor += 1
            D_neighbor_ratios.append(cosponsor / top_k)
        Dem_ratio = np.mean(D_neighbor_ratios)
        print(Dem_ratio)

        R_neighbor_ratios = []
        for query_id, neighborhood in GOP_neighborood.items():
            query = self.id_to_phrase[query_id]
            cosponsor = 0
            for neighbor_id, cosine_sim in neighborhood:
                neighbor = self.id_to_phrase[neighbor_id]
                if neighbor == query:
                    continue
                if neighbor.R_ratio > threshold and query.R_ratio > threshold:
                    cosponsor += 1
            R_neighbor_ratios.append(cosponsor / top_k)
        GOP_ratio = np.mean(R_neighbor_ratios)
        print(GOP_ratio)


def main() -> None:
    model = Embedding(
        embed_path='../../data/pretrained_word2vec/for_real.txt',
        party_freq_path='../../data/processed/plain_text/for_real/vocab_partisan_frequency.tsv',
        min_freq=100)

    Dem_ids = []
    GOP_ids = []
    very_neutral_ids = []
    kinda_neutral_ids = []
    partisan_lower_bound = 0.8
    neutral_bound = 0.1
    neutral_upper_bound = 0.5 + neutral_bound
    neutral_lower_bound = 0.5 - neutral_bound
    for phrase_id, phrase in model.id_to_phrase.items():
        if phrase.D_ratio > partisan_lower_bound:
            Dem_ids.append(phrase_id)
        elif phrase.R_ratio > partisan_lower_bound:
            GOP_ids.append(phrase_id)
        elif neutral_lower_bound < phrase.R_ratio < neutral_upper_bound:
            very_neutral_ids.append(phrase_id)
        else:
            kinda_neutral_ids.append(phrase_id)

    print(f'{len(GOP_ids)} capitalists\n'
          f'{len(Dem_ids)} socialists\n'
          f'{len(very_neutral_ids)} swing voters\n'
          f'{len(kinda_neutral_ids)} neoliberal shills')
    Dem_ids, GOP_ids, very_neutral_ids, kinda_neutral_ids = map(
        np.array, (Dem_ids, GOP_ids, very_neutral_ids, kinda_neutral_ids))

    base_dir = '../../data/evaluation/'

    # model.export_combined_nearest_partisan_neighbors(
    #     base_dir + 'Dem_sample.tsv',
    #     Dem_ids,
    #     [Dem_ids, GOP_ids, very_neutral_ids, kinda_neutral_ids])
    # model.export_combined_nearest_partisan_neighbors(
    #     base_dir + 'GOP_sample.tsv',
    #     GOP_ids,
    #     [Dem_ids, GOP_ids, very_neutral_ids, kinda_neutral_ids])

    # Dem_phrase_pairs = model.export_combined_nearest_partisan_neighbors(
    #     Dem_ids, [Dem_ids, GOP_ids, very_neutral_ids, kinda_neutral_ids],
    #     top_k=2, export_path=None)
    # GOP_phrase_pairs = model.export_combined_nearest_partisan_neighbors(
    #     GOP_ids, [Dem_ids, GOP_ids, very_neutral_ids, kinda_neutral_ids],
    #     top_k=2, export_path=None)
    # model.export_MTurk(
    #     out_path=base_dir + 'top2_partisan_neighbors.csv',
    #     phrase_pairs=Dem_phrase_pairs + GOP_phrase_pairs,  # type: ignore
    #     shuffle=True)

    # model.export_MTurk_with_context_sentences(
    #     corpus_path='../../data/processed/plain_text/for_real/corpus.txt',
    #     out_path=base_dir + 'MTurk.csv',
    #     phrase_pairs=Dem_phrase_pairs + GOP_phrase_pairs,  # type: ignore
    #     contexts_per_phrase=2)

    model.motivation_analysis(Dem_ids, GOP_ids, threshold=.5, top_k=30)




if __name__ == '__main__':
    main()
