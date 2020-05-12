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

    # Lowercase, discard punctuations, replace numbers, deduplicate
    number = re.compile(r'\d')
    starts_with_letter = re.compile(r"^\w")
    select_punctuations = re.compile(r"[@#&:]|.com")
    norm_freq: Counter[str] = Counter()
    existed: Set[Tuple[str, ...]] = set()
    duplicates = 0
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
            if conserve_RAM:
                del sent.tokens
            # all_norm_tokens += sent.normalized_tokens
            hashable = tuple(sent.normalized_tokens)
            if hashable not in existed:
                existed.add(hashable)
            else:
                duplicates += 1

        doc.sentences = [  # Filter out duplicate sentences
            sent for sent in doc.sentences
            if tuple(sent.tokens) not in existed]
    print(f'Number of duplicate sentences = {duplicates:,}')


    UNK_filtered_freq: Counter[str] = Counter()
    for key, val in norm_freq.items():
        if val >= min_frequency:
            UNK_filtered_freq[key] = val
        else:
            UNK_filtered_freq['<UNK>'] += val
    print(f'Number of filtered unigrams = {len(UNK_filtered_freq):,}')
    print(f'Number of filtered unigrams = {len(UNK_filtered_freq):,}', file=preview)


    all_norm_tokens: List[str] = [
        nt
        for doc in corpus
        for sent in doc.sentences
        for nt in sent.normalized_tokens]

    special_tokens = {'<UNK>', '<NUM>', "n't", "n’t"}
    print('Finding bigrams...')
    bigram_finder = BigramCollocationFinder.from_words(all_norm_tokens)
    num_tokens = len(all_norm_tokens)
    bigram_finder.apply_freq_filter(min_frequency)
    stop_words = set(stopwords.words('english')).union(special_tokens)
    bigram_finder.apply_word_filter(lambda word: word in stop_words)
    bigrams = bigram_finder.score_ngrams(BigramAssocMeasures().raw_freq)
    # bigrams = bigram_finder.score_ngrams(BigramAssocMeasures().pmi)
    print(f'Number of filtered bigrams = {len(bigrams):,}')
    print(f'Number of filtered bigrams = {len(bigrams):,}', file=preview)
    with open(out_dir / 'bigrams.txt', 'w') as bigram_file:
        for bigram, relative_freq in bigrams:
            absolute_freq = relative_freq * num_tokens
            bigram_str = ' '.join(bigram)
            # bigram_file.write(f'{relative_freq:.4f}\t{bigram_str}\n')  # for PMI
            bigram_file.write(f'{absolute_freq:.0f}\t{bigram_str}\n')

    # print('Finding trigrams...')
    # trigram_finder = TrigramCollocationFinder.from_words(all_norm_tokens)
    # trigram_finder.apply_freq_filter(min_frequency)
    # trigram_finder.apply_word_filter(lambda word: word in stop_words)
    # # trigram_finder.apply_ngram_filter(
    # #     lambda w1, w2, w3: (w1 in stop_words) or (w3 in stop_words) or (w2 in special_tokens))
    # trigrams = trigram_finder.score_ngrams(TrigramAssocMeasures().raw_freq)
    # print(f'Number of filtered trigrams = {len(trigrams):,}')
    # print(f'Number of filtered trigrams = {len(trigrams):,}', file=preview)
    # with open(out_dir / 'trigrams.txt', 'w') as trigram_file:
    #     for trigram, relative_freq in trigrams:
    #         absolute_freq = relative_freq * num_tokens
    #         trigram_str = ' '.join(trigram)
    #         trigram_file.write(f'{absolute_freq:.0f}\t{trigram_str}\n')
    del all_norm_tokens

    # Multi-Word Expression tokenize to underscored
    underscorer = MWETokenizer([bi for bi, _ in bigrams])  # maybe add affordable care act
    # underscorer = MWETokenizer(
    #     [tri for tri, _ in trigrams] + [bi for bi, _ in bigrams])
    vocab: Counter[str] = Counter()
    for doc in tqdm(corpus, desc='Underscoring multi-phrase expressions'):
        for sent in doc.sentences:
            sent.underscored_tokens = underscorer.tokenize(sent.normalized_tokens)
            vocab.update(sent.underscored_tokens)
            if conserve_RAM:
                del sent.normalized_tokens
    print('Pickling...')
    with open(out_dir / 'MWE_underscored.pickle', 'wb') as out_file:
        pickle.dump(corpus, out_file)

    for key, val in vocab.most_common():
        if val >= min_frequency:
            print(f'{val:,}:\t{key}', file=preview)
    preview.close()


if __name__ == '__main__':
    main(
        in_dir=Path('../../data/interim/stanza/validation'),
        out_dir=Path('../../data/interim/news/'),
        min_frequency=30,
        num_corpus_chunks=100,
        conserve_RAM=False)
