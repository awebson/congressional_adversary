import re
import pickle
from pathlib import Path
from typing import Set, Tuple, List, Dict, Counter, Iterable, Optional

from nltk.tokenize import MWETokenizer
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
# from nltk.collocations import TrigramCollocationFinder, TrigramAssocMeasures
from nltk.corpus import stopwords
from tqdm import tqdm

from data import Sentence, LabeledDoc


def main(
        in_dir: Path,
        out_dir: Path,
        num_corpus_chunks: int,
        min_frequency: int,
        conserve_RAM: bool = False
        ) -> None:
    Path.mkdir(out_dir, parents=True, exist_ok=True)
    preview = open(out_dir / f'vocab.txt', 'w')

    corpus: List[LabeledDoc] = []
    for part_index in tqdm(range(num_corpus_chunks), desc='Loading cache'):
        with open(in_dir / f'tokenized_{part_index}.pickle', 'rb') as in_file:
            corpus += pickle.load(in_file)

    existed: Set[Tuple[str, ...]] = set()
    duplicates = Counter()
    # for doc in tqdm(corpus, desc='Deduplicating'):
    #     for sent in doc.sentences:
    #         stuff = tuple(sent.tokens)
    #         if stuff not in existed:
    #             existed.add(stuff)
    #         else:
    #             duplicates[stuff] += 1
    # print(len(duplicates))
    # print(duplicates.most_common())
    # import IPython
    # IPython.embed()


    # Lowercase, discard punctuations, replace numbers
    number = re.compile(r'\d')
    starts_with_letter = re.compile(r"^\w")
    select_punctuations = re.compile(r"[@#&:]|.com")
    norm_freq: Counter[str] = Counter()
    all_norm_tokens: List[str] = []
    for doc in tqdm(corpus, desc='Normalizing tokens'):
        for sent in doc.sentences:
            for token in sent.tokens:
                if not starts_with_letter.search(token):
                    continue
                if select_punctuations.search(token):
                    continue
                if number.search(token):
                    norm_token = '<NUM>'
                else:
                    norm_token = token.lower()
                sent.normalized_tokens.append(norm_token)
                norm_freq[norm_token] += 1

            hashable = tuple(sent.normalized_tokens)
            if hashable not in existed:
                existed.add(hashable)
            else:
                duplicates[hashable] += 1

            if conserve_RAM:
                del sent.tokens
            all_norm_tokens += sent.normalized_tokens

    print(len(duplicates))
    for k, v in duplicates.most_common():
        print(v, ' '.join(k))


    # UNK_filtered_freq: Counter[str] = Counter()
    # for key, val in norm_freq.items():
    #     if val >= min_frequency:
    #         UNK_filtered_freq[key] = val
    #     else:
    #         UNK_filtered_freq['<UNK>'] += val
    # print(f'Number of filtered unigrams = {len(UNK_filtered_freq):,}')
    # print(f'Number of filtered unigrams = {len(UNK_filtered_freq):,}', file=preview)


    # # Multi-Word Expression tokenize to underscored
    # underscorer = MWETokenizer([bi for bi, _ in bigrams])  # maybe add affordable care act
    # # underscorer = MWETokenizer(
    # #     [tri for tri, _ in trigrams] + [bi for bi, _ in bigrams])
    # vocab: Counter[str] = Counter()
    # for doc in tqdm(corpus, desc='Underscoring multi-phrase expressions'):
    #     for sent in doc.sentences:
    #         sent.underscored_tokens = underscorer.tokenize(sent.normalized_tokens)
    #         vocab.update(sent.underscored_tokens)
    #         if conserve_RAM:
    #             del sent.normalized_tokens
    # with open(out_dir / 'MWE_underscored.pickle', 'wb') as out_file:
    #     pickle.dump(corpus, out_file)

    # for key, val in vocab.most_common():
    #     if val >= min_frequency:
    #         print(f'{val:,}:\t{key}', file=preview)
    # preview.close()


if __name__ == '__main__':
    main(
        in_dir=Path('../../data/interim/stanza/validation'),
        out_dir=Path('../../data/interim/news'),
        # in_dir=Path('../../data/interim/news/train'),
        # out_dir=Path('../../data/interim/news/train_pmi'),
        min_frequency=30,
        num_corpus_chunks=100,
        conserve_RAM=True)
