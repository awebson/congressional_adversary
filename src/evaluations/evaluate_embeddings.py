import pickle
import os
from collections import namedtuple
from typing import Set, Tuple, List, Dict

import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise
from nltk.corpus import wordnet
from tqdm import tqdm

np.random.seed(1)
sns.set()

# def load_embedding(path: str):
    # from pretrain_embeddings import SkipGramNegativeSampling
    # from embeddings.party_classifier import NaivePartyDiscriminator
    # model = NaivePartyDiscriminator(
    #     vocab_size=len(id_to_word), embed_size=300)
    # model.load_state_dict(torch.load(model_path))


def export_embedding(out_path: str) -> str:
    """export embeddings in plain text for external evaluation"""
    pass


def load_plain_text_embeddings(path: str) -> Tuple[np.array, Dict[str, int]]:
    """TODO turn off auto_caching"""
    cache_path = path + '.pickle'
    if os.path.exists(cache_path):
        print(f'Loading cache from {cache_path}', end='\t', flush=True)
        with open(cache_path, 'rb') as cache_file:
            return pickle.load(cache_file)
        print('Done')
    else:
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
        with open(cache_path, 'wb') as cache_file:
            print(f'Caching to {cache_path}', end='\t', flush=True)
            pickle.dump(
                (np.array(embeddings), word_to_id), cache_file, protocol=4)
        print('Done')
        return np.array(embeddings), word_to_id


def export_fasttext_tensorboard_projector() -> None:
    embeddings, word_to_id = load_fasttext_embedding('embeddings/GOP_model.vec')
    # base_path = 'embeddings/GOP_model'
    # with open(base_path + '_tensorboard.tsv', 'w') as vector_file:
    #     for vector in embeddings:
    #         vector_file.write('\t'.join(map(str, vector)) + '\n')

    # id_to_word = {val: key for key, val in word_to_id.items()}
    # with open(base_path + '_tensorboard_labels.tsv', 'w') as label_file:
    #     for index in range(len(word_to_id)):
    #         label_file.write(id_to_word[index] + '\n')

    base_path = 'embeddings/GOP_model_subset'
    random_indicies = np.random.randint(len(embeddings), size=10000)
    random_embeddings = embeddings[random_indicies]
    with open(base_path + '_tensorboard.tsv', 'w') as vector_file:
        for vector in random_embeddings:
            vector_file.write('\t'.join(map(str, vector)) + '\n')

    id_to_word = {val: key for key, val in word_to_id.items()}
    with open(base_path + '_tensorboard_labels.tsv', 'w') as label_file:
        for index in random_indicies:
            label_file.write(id_to_word[index] + '\n')


def evaluate_synonyms_antonyms(
        embeddings: np.array,
        word_to_id: Dict[str, int],
        graph_path: str
        ) -> None:
    """filters only those in embedding vocabulary"""

    def mean_pairwise_similarity(
            query_indicies: List[int],
            reduce: bool = True
            ) -> np.array:
        queried_embeddings = embeddings[query_indicies]
        distances = pairwise.cosine_similarity(queried_embeddings)
        upper_traingular = np.triu_indices_from(distances, k=1)
        if reduce:
            return np.mean(distances[upper_traingular])
        else:
            return distances[upper_traingular]

    wordnet_count: int = 0
    in_vocab_count: int = 0
    synonym_distances = []
    antonym_distances = []
    for synset in tqdm(wordnet.all_synsets(),
                       desc='WordNet Synsets', total=117659):
        synonyms: List[int] = []
        for lemma in synset.lemmas():
            wordnet_count += 1
            word_text = lemma.name()
            if word_text in word_to_id:
                # synonyms.append(word_text)
                synonyms.append(word_to_id[word_text])
                in_vocab_count += 1

                if lemma.antonyms():
                    # antonyms = [ant.name()
                    #             for ant in lemma.antonyms()
                    #             if ant.name() in word_to_id]
                    # print(word_text, antonyms)
                    antonyms = [word_to_id[ant.name()]
                                for ant in lemma.antonyms()
                                if ant.name() in word_to_id]
                    # might be an empty list due to out of vocabulary
                    if len(antonyms) > 1:
                        antonyms.append(word_to_id[word_text])
                        antonym_distances.append(
                            mean_pairwise_similarity(antonyms))
        # print('')
        if len(synonyms) > 1:
            # print(synonyms)
            synonym_distances.append(mean_pairwise_similarity(synonyms))

    random_indicies = np.random.randint(len(embeddings), size=10000)
    random_distances = mean_pairwise_similarity(random_indicies, reduce=False)
    print(f'Corpus Vocab / WordNet Vocab Ratio = {in_vocab_count / wordnet_count:.4%}')
    print(f'Synonyms mean similarity = {np.mean(synonym_distances):.4}')
    print(f'Antonyms mean similarity = {np.mean(antonym_distances):.4}')
    print(f'Random mean similarity = {np.mean(random_distances):.4}')
    print(f'Plotting...')

    sns.distplot(random_distances, label='random')
    sns.distplot(antonym_distances, label='antonyms')
    sns.distplot(synonym_distances, label='synonyms')

    plt.legend()
    plt.savefig(graph_path)


def write_all_word_partisan_connotation(
        model_path: str,
        id_to_word: Dict[int, str]
        ) -> None:
    # TODO clean up IO
    from party_discriminator import NaivePartyDiscriminator
    model = NaivePartyDiscriminator(
        vocab_size=len(id_to_word), embed_size=300)
    model.load_state_dict(torch.load(model_path))

    all_word_ids = np.arange(len(id_to_word))
    confidences = model.evaluate(all_word_ids)

    Output = namedtuple('Output', ['word', 'socialism', 'capitalism']) # 'prediction'])
    write_to_file: List[Output] = []
    for word_id, confidence in enumerate(confidences):
        output = Output(
            id_to_word[word_id], confidence[0].item(), confidence[1].item())
        # if output.capitalism > .85:
        #     output.prediction = 'galaxy brain: regressive tax is good'
        # elif .7 < output.capitalism <= .85:
        #     output.prediction = 'entitlement reform anyone?'
        # elif .5 < output.capitalism <= .7:
        #     output.prediction = 'Susan Collins et al. have the best seats'
        # elif .5 < output.socialism <= .7:
        #     output.prediction = 'not crazy about getting rid of the filibuster'
        # elif .7 < socialism < .85:
        #     output.prediction = 'neoliberal shill'
        # else:
        #     output.prediction ='political revolution'
        write_to_file.append(output)

    # write_to_file.sort(key=lambda tupl: tupl.socialism)
    # with open('evaluation/party_prediction.txt', 'w') as eval_file:
    #     eval_file.write(f'Socialism v. Capitalism\tWord\tPrediction\n')
    #     for output in write_to_file:
    #         eval_file.write(
    #             f'{output.socialism:.3f}, {output.capitalism:.3f}\t\t'
    #             f'{output.word}\n')

    # political_revolution = [out.socialism for out in write_to_file]
    tax_cuts_grow_the_economy = [out.capitalism for out in write_to_file]
    # sns.distplot(political_revolution, label='socialism')
    sns.distplot(tax_cuts_grow_the_economy, label='capitalism')
    plt.legend()
    plt.savefig('graphs/capital_distribution.pdf')


# def main() -> None:
#     model_path = '../models/skip_gram/paper1e-5/epoch0.pt'
#     # model_path = '../models/skip_gram/1e-3/epoch0.pt'
#     with open(model_path, 'rb') as model_file:
#         state_dict = torch.load(model_file, map_location='cpu')
#     # print(state_dict.keys())
#     embeddings = state_dict['center_embedding.weight'].numpy()

#     vocab_path = '../corpora/skip_grams/paper1e-5_10chunks/vocab.pickle'
#     # vocab_path = '../corpora/skip_grams/1e-3_10chunk/vocab.pickle'
#     with open(vocab_path, 'rb') as vocab_file:
#         word_to_id, id_to_word, _ = pickle.load(vocab_file)


#     # embeddings, word_to_id = load_plain_text_embeddings('../models/baseline/fastText.txt')

#     # evaluate_synonyms_antonyms(embeddings, word_to_id, '1e-5.pdf')

#     # write_all_word_partisan_connotation(model_path, id_to_word)
#     # export_tensorboard_projector(model_path, id_to_word)

#     # random_indicies = np.random.randint(len(embeddings), size=10000)
#     # random_embeddings = embeddings[random_indicies]
#     # distances = pairwise.cosine_similarity(random_embeddings)

#     # query_pair = ('undocumented_immigrants', 'illegal_aliens')
#     query_pair = ('piano', 'blueberry')
#     query_indicies = tuple(map(word_to_id.get, query_pair))
#     if None in query_indicies:
#         raise KeyError(f'Out of vocabulary. Sorry!')
#     query_vectors = embeddings[query_indicies, :]
#     similarity = 1 - scipy.spatial.distance.cosine(query_vectors[0], query_vectors[1])
#     print(similarity)


if __name__ == '__main__':
    from pretrain_embeddings import SkipGramNegativeSampling
    # main()
