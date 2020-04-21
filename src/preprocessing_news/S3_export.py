import re
import math
import random
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, DefaultDict, Counter, Optional

from tqdm import tqdm

from preprocessing_news.S1_tokenize import Sentence, LabeledDoc

random.seed(42)


def build_vocabulary(
        frequency: Counter
        ) -> Tuple[
        Dict[str, int],
        Dict[int, str]]:
    word_to_id: Dict[str, int] = {}
    id_to_word: Dict[int, str] = {}
    word_to_id['<PAD>'] = 0
    word_to_id['<UNK>'] = 1
    id_to_word[0] = '<PAD>'
    id_to_word[1] = '<UNK>'

    next_vocab_id = 2
    for word, freq in frequency.items():
        if word not in word_to_id:
            word_to_id[word] = next_vocab_id
            id_to_word[next_vocab_id] = word
            next_vocab_id += 1
    print(f'Vocabulary size = {len(word_to_id):,}')
    return word_to_id, id_to_word


def subsampling(
        frequency: Counter[str],
        heuristic: Optional[str],
        threshold: float,
        ) -> Dict[str, float]:
    """
    Downsample frequent words.

    Subsampling implementation from annotated C code of Mikolov et al. 2013:
    http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling
    This blog post is linked from TensorFlow's website, so authoratative?

    NOTE the default threshold is 1e-3, not 1e-5 as in the paper version
    """
    cumulative_freq = sum(abs_freq for abs_freq in frequency.values())
    keep_prob: Dict[str, float] = dict()

    if heuristic is None:
        from typing import DefaultDict
        keep_prob = DefaultDict(lambda: 1)
        return keep_prob

    if heuristic == 'code':
        for word, abs_freq in frequency.items():
            rel_freq = abs_freq / cumulative_freq
            keep_prob[word] = (
                (math.sqrt(rel_freq / threshold) + 1)
                * (threshold / rel_freq)
            )
    elif heuristic == 'paper':
        for word, abs_freq in frequency.items():
            rel_freq = abs_freq / cumulative_freq
            keep_prob[word] = math.sqrt(threshold / rel_freq)
    else:
        raise ValueError('Unknown heuristic of subsampling.')
    return keep_prob


def main(
        in_dir: Path,
        out_dir: Path,
        num_corpus_chunks: int,
        min_frequency: int,
        min_sent_len: int,
        max_sent_len: int,
        subsample_heuristic: Optional[str],
        subsample_threshold: float,
        conserve_RAM: bool = True  # turn off to inspect intermediate results
        ) -> None:
    corpus: List[LabeledDoc] = []
    for part_index in tqdm(range(num_corpus_chunks), desc='Loading cache'):
        with open(in_dir / f'tokenized_{part_index}.pickle', 'rb') as in_file:
            corpus += pickle.load(in_file)

    Path.mkdir(out_dir, parents=True, exist_ok=True)
    preview = open(out_dir / f'preview.txt', 'w')
    print(f'Min word frequency = {min_frequency}', file=preview)
    print(f'Min sentence length = {min_sent_len}', file=preview)
    print(f'Max sentence length = {max_sent_len}', file=preview)
    print(f'SGNS subsample heuristic= {subsample_heuristic}', file=preview)
    print(f'SGNS subsample threshold = {subsample_threshold}', file=preview)

    # Lowercase, discard punctuations, replace numbers
    number = re.compile(r'\d')
    starts_with_letter = re.compile(r"^\w")
    nonalphanumeric = re.compile(r"[^a-zA-Z0-9_.\-'’]")  # allow in-word dash or period
    # mystery_ellipsis = re.compile(r"[.]+2")
    norm_freq: Counter[str] = Counter()
    for doc in tqdm(corpus, desc='Normalize tokens'):
        for sent in doc.sentences:
            for token in sent.underscored_tokens:  # NOTE cf. .tokens
                if not starts_with_letter.search(token):
                    continue
                # if nonalphanumeric.search(token):
                #     continue
                if number.search(token):
                    norm_token = '<NUM>'
                else:
                    norm_token = token.lower()
                sent.normalized_tokens.append(norm_token)
                norm_freq[norm_token] += 1
            if conserve_RAM:
                del sent.tokens
    print(f'Noramlized vocabulary size = {len(norm_freq):,}', file=preview)
    print(f'Presampled number of words = '
          f'{sum(freq for freq in norm_freq.values()):,}',
          file=preview)

    # Filter counter with MIN_FREQ and count UNK
    new_freq: Counter[str] = Counter()
    for key, val in norm_freq.items():
        if val >= min_frequency:
            new_freq[key] = val
        else:
            new_freq['<UNK>'] += val
    norm_freq = new_freq
    print(f'Filtered vocabulary size = {len(norm_freq):,}', file=preview)

    # Count connotation grounding prior to subsampling trick
    numericalize_cono = {
        'left': 0,
        'left-center': 1,
        'least': 2,
        'right-center': 3,
        'right': 4}
    # cono_grounding: DefaultDict[str, List[int]] = DefaultDict(
    #     lambda: [0, 0, 0, 0, 0])  # unfortunately unpicklable
    cono_grounding: Dict[str, List] = {'<PAD>': [0, 0, 0, 0, 0]}
    # Subsampling & filter by mix/max sentence length
    keep_prob = subsampling(norm_freq, subsample_heuristic, subsample_threshold)
    final_freq: Counter[str] = Counter()
    for doc in tqdm(corpus, desc='Ground connotation & subsample frequent words'):
        for sent in doc.sentences:
            for token in sent.normalized_tokens:
                if token not in norm_freq:
                    token = '<UNK>'
                if random.random() < keep_prob[token]:
                    sent.subsampled_tokens.append(token)
                if token not in cono_grounding:
                    cono_grounding[token] = [0, 0, 0, 0, 0]
                cono_grounding[token][numericalize_cono[doc.party]] += 1
            # End looping tokens
            if len(sent.subsampled_tokens) >= min_sent_len:
                if len(sent.subsampled_tokens) <= max_sent_len:
                    final_freq.update(sent.subsampled_tokens)
                else:  # NOTE truncate long sentences
                    sent.subsampled_tokens = sent.subsampled_tokens[:max_sent_len]
                    final_freq.update(sent.subsampled_tokens)
            else:
                sent.subsampled_tokens = None
            if conserve_RAM:
                del sent.normalized_tokens
        # End looping sentences
        doc.sentences = [
            sent for sent in doc.sentences
            if sent.subsampled_tokens is not None]
    # End looping documents
    corpus = [doc for doc in corpus if len(doc.sentences) > 0]

    print(f'Final vocabulary size = {len(final_freq):,}', file=preview)
    print(f'Subsampled number of words = '
          f'{sum(freq for freq in final_freq.values()):,}',
          file=preview)

    # Numericalize corpus by word_ids
    word_to_id, id_to_word = build_vocabulary(final_freq)
    for doc in tqdm(corpus, desc='Converting to word ids'):
        for sent in doc.sentences:
            sent.numerical_tokens = [
                word_to_id[token] for token in sent.subsampled_tokens]
            if conserve_RAM:
                del sent.subsampled_tokens

    # Helper for negative sampling
    cumulative_freq = sum(freq ** 0.75 for freq in final_freq.values())
    negative_sampling_probs: Dict[int, float] = {
        word_to_id[word]: (freq ** 0.75) / cumulative_freq
        for word, freq in final_freq.items()}
    negative_sampling_probs: List[float] = [
        # negative_sampling_probs[word_id]  # strict
        negative_sampling_probs.get(word_id, 0)  # prob = 0 if missing vocab
        for word_id in range(len(word_to_id))]

    random.shuffle(corpus)
    cucumbers = {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'cono_grounding': cono_grounding,
        'negative_sampling_probs': negative_sampling_probs,
        'documents': corpus}
    print(f'Writing to {out_dir}')
    with open(out_dir / 'train.pickle', 'wb') as out_file:
        pickle.dump(cucumbers, out_file, protocol=-1)

    # Print out vocabulary & some random sentences for sanity check
    docs = random.sample(corpus, 100)
    preview.write('\n')
    for doc in docs:
        sent = doc.sentences[0]
        if not conserve_RAM:
            print(sent.tokens, file=preview)
            print(sent.normalized_tokens, file=preview)
            print(sent.numerical_tokens, file=preview)
            print(sent.subsampled_tokens, end='\n\n', file=preview)
        else:
            print(sent.numerical_tokens, file=preview)
            # print(vars(doc), end='\n\n', file=preview)
    preview.write('\n')
    for key, val in final_freq.most_common():
        print(f'{val:,}:\t{key}\t{cono_grounding[key]}', file=preview)
    preview.close()
    print('All set!')


if __name__ == '__main__':
    main(
        in_dir=Path('../../data/interim/news'),
        out_dir=Path('../../data/processed/news/toy'),
        min_frequency=10,
        min_sent_len=5,
        max_sent_len=20,
        num_corpus_chunks=3,
        subsample_heuristic='paper',
        subsample_threshold=1e-3,
        conserve_RAM=False)
