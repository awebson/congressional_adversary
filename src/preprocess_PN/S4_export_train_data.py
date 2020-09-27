import math
import random
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Counter, Iterable, Optional

import numpy as np
from tqdm import tqdm

from data import Sentence, LabeledDoc, GroundedWord

random.seed(42)


def build_vocabulary(
        frequency: Counter,
        min_frequency: int = 0,
        add_special_tokens: bool = True
        ) -> Tuple[
        Dict[str, int],
        Dict[int, str]]:
    word_to_id: Dict[str, int] = {}
    if add_special_tokens:
        word_to_id['[PAD]'] = 0
        word_to_id['[UNK]'] = 1
        word_to_id['[CLS]'] = 2
        word_to_id['[SEP]'] = 3
    id_to_word = {val: key for key, val in word_to_id.items()}
    next_vocab_id = len(word_to_id)
    for word, freq in frequency.items():
        if word not in word_to_id and freq >= min_frequency:
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
        min_frequency: int,
        min_sent_len: int,
        max_sent_len: int,
        subsample_heuristic: Optional[str],
        subsample_threshold: float,
        eval_min_freq: int,
        eval_R_thresholds: Iterable[float],
        eval_num_random_samples: int,
        conserve_RAM: bool = True  # turn off to inspect intermediate results
        ) -> None:
    Path.mkdir(out_dir, parents=True, exist_ok=True)
    preview = open(out_dir / f'preview.txt', 'w')
    print(f'Min word frequency = {min_frequency}', file=preview)
    print(f'Min sentence length = {min_sent_len}', file=preview)
    print(f'Max sentence length = {max_sent_len}', file=preview)
    print(f'SGNS subsample heuristic= {subsample_heuristic}', file=preview)
    print(f'SGNS subsample threshold = {subsample_threshold}', file=preview)

    corpus: List[LabeledDoc] = []
    print('Loading multi-word expression underscored pickle...')
    with open(in_dir / f'MWE_underscored.pickle', 'rb') as in_file:
        corpus += pickle.load(in_file)

    norm_freq: Counter[str] = Counter()
    for doc in tqdm(corpus, desc='Counting UNKs'):
        if conserve_RAM:
            doc.text = None
        for sent in doc.sentences:
            norm_freq.update(sent.underscored_tokens)
    cumulative_freq = sum(freq for freq in norm_freq.values())
    print(f'Noramlized vocabulary size = {len(norm_freq):,}', file=preview)
    print(f'Number of words = {cumulative_freq:,}', file=preview)

    # Filter counter with MIN_FREQ and count UNK
    UNK_filtered_freq: Counter[str] = Counter()
    for key, val in norm_freq.items():
        if val >= min_frequency:
            UNK_filtered_freq[key] = val
        else:
            UNK_filtered_freq['[UNK]'] += val
    print(f'Filtered vocabulary size = {len(UNK_filtered_freq):,}', file=preview)
    assert sum(freq for freq in norm_freq.values()) == cumulative_freq


    # Subsampling & filter by min/max sentence length
    ground: Dict[str, GroundedWord] = {}
    keep_prob = subsampling(
        UNK_filtered_freq, subsample_heuristic, subsample_threshold)
    final_freq: Counter[str] = Counter()
    for doc in tqdm(corpus, desc='Subsampling frequent words'):
        for sent in doc.sentences:
            for token in sent.underscored_tokens:
                if token not in UNK_filtered_freq:
                    token = '[UNK]'
                if random.random() < keep_prob[token]:
                    sent.subsampled_tokens.append(token)
            # End looping tokens

            if len(sent.subsampled_tokens) >= min_sent_len:
                if len(sent.subsampled_tokens) > max_sent_len:
                    sent.subsampled_tokens = sent.subsampled_tokens[:max_sent_len]
                final_freq.update(sent.subsampled_tokens)
                for word in sent.subsampled_tokens:
                    if word not in ground:
                        ground[word] = GroundedWord(
                            text=word, deno=None, cono=Counter({doc.party: 1}))
                    else:
                        ground[word].cono[doc.party] += 1
            else:  # discard short sentences
                sent.subsampled_tokens = None
            if conserve_RAM:
                sent.tokens = None
                sent.normalized_tokens = None
                sent.underscored_tokens = None
        # End looping sentences

        doc.sentences = [  # Filter out empty sentences
            sent for sent in doc.sentences
            if sent.subsampled_tokens is not None]
    # End looping documents
    print(f'Final vocabulary size = {len(final_freq):,}', file=preview)
    print(f'Subsampled number of words = '
          f'{sum(freq for freq in final_freq.values()):,}', file=preview)

    # Filter out empty documents
    corpus = [doc for doc in corpus if len(doc.sentences) > 0]

    # Numericalize corpus by word_ids
    word_to_id, id_to_word = build_vocabulary(final_freq)
    for doc in tqdm(corpus, desc='Converting to word ids'):
        for sent in doc.sentences:
            sent.numerical_tokens = [
                word_to_id[token] for token in sent.subsampled_tokens]
            if conserve_RAM:
                sent.subsampled_tokens = None

    # # Compute PMI
    # def prob(count: int) -> float:
    #     return count / cumulative_freq  # presampled frequency

    # Prepare grounding for intrinsic evaluation
    random_eval_words = set()
    for gw in ground.values():
        gw.majority_cono = gw.cono.most_common(1)[0][0]
        gw.freq = sum(gw.cono.values())
        gw.R_ratio = (gw.cono['right'] + gw.cono['right-center']) / gw.freq
        if gw.freq >= eval_min_freq:
            random_eval_words.add(gw.text)
    random_eval_words = random.sample(random_eval_words, eval_num_random_samples)
    with open(out_dir / f'eval_words_random.txt', 'w') as file:
        file.write('\n'.join(random_eval_words))

    for R_threshold in eval_R_thresholds:
        D_threshold = 1 - R_threshold
        partisan_eval_words = []
        for gw in ground.values():
            if gw.freq >= eval_min_freq:
                if gw.R_ratio >= R_threshold or gw.R_ratio <= D_threshold:
                    partisan_eval_words.append(gw)
        print(f'{len(partisan_eval_words)} partisan eval words '
              f'with R_threshold = {R_threshold}', file=preview)

        out_path = out_dir / f'inspect_{R_threshold}_partisan.tsv'
        with open(out_path, 'w') as file:
            print('word\tfreq\tR_ratio', file=file)
            for gw in partisan_eval_words:
                print(gw.text, gw.freq, gw.R_ratio, sep='\t', file=file)

        if len(partisan_eval_words) > 2 * eval_num_random_samples:
            partisan_eval_words = random.sample(
                partisan_eval_words, 2 * eval_num_random_samples)
        else:
            random.shuffle(partisan_eval_words)

        mid = len(partisan_eval_words) // 2
        with open(out_dir / f'{R_threshold}partisan_dev_words.txt', 'w') as file:
            for gw in partisan_eval_words[:mid]:
                print(gw.text, file=file)
        with open(out_dir / f'{R_threshold}partisan_test_words.txt', 'w') as file:
            for gw in partisan_eval_words[mid:]:
                print(gw.text, file=file)

    # ground: Dict[str, GroundedWord] = {}
    # # cono_labels = set(numericalize_cono.values())
    # for word in tqdm(word_to_id.keys(), desc='Computing PMIs (-∞ are okay)'):
    #     cono = np.array(cono_freq[word])
    #     # cono_ratio = cono / np.sum(cono)
    #     # PMI = np.log2([  # can be -inf if freq = 0
    #     #     prob(cono[party_id])
    #     #     / (prob(norm_freq[word]) * prob(party_cumulative[party_id]))
    #     #     for party_id in cono_labels])
    #     ground[word] = GroundedWord(text=word, deno=None, cono=cono)

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
        'ground': ground,
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
            print(sent.subsampled_tokens, file=preview)
            print(sent.numerical_tokens, file=preview, end='\n\n')
        else:
            print(sent.numerical_tokens, file=preview)
            # print(vars(doc), end='\n\n', file=preview)
    preview.write('\n\nfinal_freq\tword\n')
    for key, val in final_freq.most_common():
        print(f'{val:,}\t{ground[key]}', file=preview)
    preview.close()
    print('All set!')


if __name__ == '__main__':
    main(
        in_dir=Path('../../data/interim/news'),
        out_dir=Path('../../data/ready/PN_proxy'),
        min_frequency=30,
        min_sent_len=5,
        max_sent_len=20,
        subsample_heuristic='paper',
        subsample_threshold=1e-5,
        eval_min_freq=100,
        eval_R_thresholds=(0.6, 0.7, 0.75, 0.8, 0.9),
        eval_num_random_samples=500,
        conserve_RAM=True)
