import pickle
import random
import math
from pathlib import Path
from typing import Tuple, List, Dict, DefaultDict, Counter, Iterable, Optional

from nltk.corpus import stopwords
from tqdm import tqdm

from data import GroundedWord, Sentence, LabeledDoc

procedural_words = {
    'yield', 'motion', 'order', 'ordered', 'quorum', 'roll', 'unanimous',
    'mr', 'madam', 'speaker', 'chairman', 'president', 'senator',
    'gentleman', 'colleague',
    'today', 'rise', 'rise today', 'pleased_to_introduce',
    'introducing_today', 'would_like'
}
discard = set(stopwords.words('english')).union(procedural_words)
random.seed(1)

# punctuations = '!"#$%&\'()*+,-—./:;<=>?@[\\]^`{|}~'  # excluding underscore
# remove_punctutation = str.maketrans('', '', punctuations)
# remove_numbers = str.maketrans('', '', '0123456789')

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
        keep_prob = DefaultDict(lambda: 1)
        return keep_prob

    if heuristic == 'code':
        for word_id, abs_freq in frequency.items():
            rel_freq = abs_freq / cumulative_freq
            keep_prob[word_id] = (
                (math.sqrt(rel_freq / threshold) + 1)
                * (threshold / rel_freq)
            )
    elif heuristic == 'paper':
        for word_id, abs_freq in frequency.items():
            rel_freq = abs_freq / cumulative_freq
            keep_prob[word_id] = math.sqrt(threshold / rel_freq)
    else:
        raise ValueError('Unknown heuristic of subsampling.')
    return keep_prob


def partition(speeches: List, num_chunks: int) -> Iterable[List]:
    chunk_size = len(speeches) // num_chunks
    speech_index = 0
    chunk_index = 0
    while chunk_index <= num_chunks - 2:
        yield speeches[speech_index:speech_index + chunk_size]
        speech_index += chunk_size
        chunk_index += 1
    yield speeches[speech_index:-1]


def export_sorted_frequency(
        raw_frequency: Counter[str],
        subsampled_frequency: Counter[str],
        min_freq: int,
        out_path: str
        ) -> None:
    output_iterable: List[Tuple[int, int, str]] = []
    for word, raw_freq in raw_frequency.items():
        if raw_freq > min_freq:
            try:
                final_freq = subsampled_frequency[word]
            except KeyError:
                final_freq = 0
            output_iterable.append((raw_freq, final_freq, word))
    output_iterable.sort(key=lambda tup: tup[0], reverse=True)
    with open(out_path, 'w') as out_file:
        for raw_freq, final_freq, phrase in output_iterable:
            out_file.write(f'{raw_freq:,}\t{final_freq:,}\t{phrase}\n')


def export_sampled_frequency_by_party(
        D_raw: Counter[str],
        R_raw: Counter[str],
        D_final: Counter[str],
        R_final: Counter[str],
        word_to_id: Dict[str, int],
        out_path: str
        ) -> None:

    def sort_creteria(tup: Tuple) -> Tuple[bool, float]:
        df, rf = tup[1], tup[2]  # initial frequnecy
        if df != 0:
            ratio = rf / df
        else:
            ratio = rf / 1e-8
        nontrivial = df + rf > 100
        return nontrivial, ratio

    output = []
    for word in word_to_id:
        output.append(
            (word, D_raw[word], R_raw[word], D_final[word], R_final[word]))
    output.sort(key=sort_creteria, reverse=True)
    with open(out_path + '_pretty.txt', 'w') as out_file:
        out_file.write('Sorted by GOP/Dem Ratio    '
                       'Original (Dem, GOP)    '
                       'Sampled & Balanced [Dem, GOP]\n')
        for word, dr, rr, df, rf in output:
            raw_freq = f'({dr:,}, {rr:,})'
            final_freq = f'[{df:,}, {rf:,}]'
            out_file.write(f'{word:<30}{raw_freq:<20}{final_freq}\n')

    with open(out_path + '.tsv', 'w') as out_file:
        out_file.write('D/R_Ratio\tTotal_Freq\tD_Freq\tR_Freq\tPhrase\n')
        for word, dr, rr, df, rf in output:
            try:
                ratio = dr / rr
            except ZeroDivisionError:
                ratio = float('inf')
            out_file.write(f'{ratio:.5}\t{dr + rr}\t{dr}\t{rr}\t{word}\n')


# def export_plain_text_corpus(
#         sessions: Iterable[int],
#         output_dir: str,
#         in_dir: str
#         ) -> None:
#     output_file = open(os.path.join(output_dir, 'corpus.txt'), 'w')
#     num_words_exported = 0
#     for session in tqdm(sessions, desc='Loading underscored corpora'):
#         for party in ('D', 'R'):
#             underscored_path = os.path.join(
#                 in_dir, f'underscored_{party}{session}.txt')
#             with open(underscored_path, 'r') as underscored_corpus:
#                 for line in underscored_corpus:
#                     num_words_exported += len(line.split())
#                     output_file.write(line)
#     output_file.close()
#     print(f'Total number of words = {num_words_exported:,}')


# def count_partisan_frequency(
#         sessions: Iterable[int],
#         in_dir: Path
#         ) -> Tuple[Counter[str], Counter[str], Counter[str]]:
#     D_freq: Counter[str] = Counter()
#     R_freq: Counter[str] = Counter()
#     for session in tqdm(sessions, desc='Counting partisan frequency'):
#         for party in ('D', 'R'):
#             underscored_path = in_dir / f'underscored_{party}{session}.txt'
#             with open(underscored_path, 'r') as underscored_corpus:
#                 for line in underscored_corpus:
#                     # line = line.translate(remove_punctutation)  # Done in S3
#                     words = line.split()
#                     if party == 'D':
#                         D_freq.update(words)
#                     else:
#                         R_freq.update(words)
#     combined_frequency = D_freq + R_freq
#     return combined_frequency, D_freq, R_freq


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


def _export_sorted_frequency_by_party(
        D_freq: Counter[str],
        R_freq: Counter[str],
        word_to_id: Dict[str, int],
        out_path: str,
        min_freq: int
        ) -> None:
    output = []
    for word in word_to_id:
        df = D_freq[word]
        rf = R_freq[word]
        total = df + rf
        above_min_freq = total > min_freq
        dr = df / total
        rr = rf / total
        output.append((above_min_freq, dr, rr, df, rf, total, word))
    output.sort(key=lambda tup: (tup[0], tup[2], tup[4]), reverse=True)

    with open(out_path, 'w') as out_file:
        out_file.write(
            f'd_ratio\tr_ratio\td_freq\tr_freq\tphrase\n')
        for above_min_freq, dr, rr, df, rf, total, phrase in output:
            out_file.write(f'{dr:.5}\t{rr:.5}\t{df}\t{rf}\t{phrase}\n')


def export_sorted_frequency_by_party(
        sessions: Iterable[int],
        output_dir: str,
        in_dir: Path,
        training_min_freq: int,
        sort_min_freq: int
        ) -> None:
    D_freq: Counter[str] = Counter()
    R_freq: Counter[str] = Counter()
    for session in tqdm(sessions, desc='Loading underscored corpora'):
        for party in ('D', 'R'):
            underscored_path = in_dir / f'underscored_{party}{session}.txt'
            with open(underscored_path, 'r') as underscored_corpus:
                for line in underscored_corpus:
                    words = line.split()
                    if party == 'D':
                        D_freq.update(words)
                    else:
                        R_freq.update(words)
    word_to_id, _ = build_vocabulary(D_freq + R_freq, training_min_freq)
    output_path = output_dir / 'vocab_partisan_frequency.tsv'
    _export_sorted_frequency_by_party(
        D_freq, R_freq, word_to_id, output_path, sort_min_freq)


def balance_classes(socialism: List, capitalism: List) -> List:
    total = len(socialism) + len(capitalism)
    S_ratio = len(socialism) / total
    C_ratio = len(capitalism) / total
    print(f'Pre-balanced Dem = {len(socialism):,}\t{S_ratio:.2%}')
    print(f'Pre-balanced GOP = {len(capitalism):,}\t{C_ratio:.2%}')
    minority = min(len(capitalism), len(socialism))
    if len(capitalism) > len(socialism):
        capitalism = random.sample(capitalism, k=minority)
        print(f'Balancing training data by sampling GOP to {minority:,}.')
    else:
        socialism = random.sample(socialism, k=minority)
        print(f'Balancing training data by sampling Dem to {minority:,}.')
    gridlock = socialism + capitalism
    return gridlock


def faux_sent_tokenize(
        tokens: List,
        fixed_sent_len: int,
        min_sent_len: int
        ) -> Iterable[List[str]]:
    """partion a document into fixed-length faux sentences"""
    start_index = 0
    while (start_index + fixed_sent_len) < (len(tokens) - 1):
        yield tokens[start_index:start_index + fixed_sent_len]
        start_index += fixed_sent_len

    trailing_words = tokens[start_index:-1]
    if len(trailing_words) >= min_sent_len:
        yield trailing_words


def main(
        sessions: Iterable[int],
        in_dir: Path,
        out_dir: Path,
        subsampling_implementation: Optional[str],
        subsampling_threshold: float,
        min_word_freq: int,
        min_sent_len: int,
        fixed_sent_len: int,
        eval_min_freq: int,
        eval_R_thresholds: Iterable[float],
        eval_num_random_samples: int,
        conserve_RAM: bool
        ) -> None:
    Path.mkdir(out_dir, parents=True, exist_ok=True)
    preview = open(out_dir / f'preview.txt', 'w')
    print(f'Reading sessions {sessions}. Writing to {out_dir}')
    print(f'Reading sessions {sessions}. Writing to {out_dir}', file=preview)
    print(f'Min word frequency = {min_word_freq}', file=preview)
    print(f'Min sentence length = {min_sent_len}', file=preview)
    print(f'Faux sentence fixed length = {fixed_sent_len}', file=preview)
    print(f'SGNS subsample implementation= {subsampling_implementation}', file=preview)
    print(f'SGNS subsample threshold = {subsampling_threshold}', file=preview)

    corpus: List[LabeledDoc] = []
    norm_freq: Counter[str] = Counter()
    for session in tqdm(
            sessions,
            desc='Loading multi-word expression underscored pickles...'):
        for party in ('D', 'R'):
            in_path = in_dir / f'underscored_{party}{session}.txt'
            with open(in_path) as underscored_corpus:
                for line in underscored_corpus:
                    underscored_tokens = line.split()
                    norm_freq.update(underscored_tokens)
                    corpus.append(LabeledDoc(
                        uid=None,
                        title=None,
                        url=None,
                        party=party,
                        referent=None,
                        text=underscored_tokens,
                        date=None,
                        sentences=[]))
    cumulative_freq = sum(freq for freq in norm_freq.values())
    print(f'Noramlized vocabulary size = {len(norm_freq):,}', file=preview)
    print(f'Number of words = {cumulative_freq:,}', file=preview)

    # Filter counter with MIN_FREQ and count UNK
    UNK_filtered_freq: Counter[str] = Counter()
    for key, val in norm_freq.items():
        if val >= min_word_freq:
            UNK_filtered_freq[key] = val
        else:
            UNK_filtered_freq['[UNK]'] += val
    print(f'Filtered vocabulary size = {len(UNK_filtered_freq):,}', file=preview)
    assert sum(freq for freq in norm_freq.values()) == cumulative_freq

    # Subsampling & filter by min/max sentence length
    keep_prob = subsampling(
        UNK_filtered_freq, subsampling_implementation, subsampling_threshold)
    ground: Dict[str, GroundedWord] = {}
    final_freq: Counter[str] = Counter()
    for doc in tqdm(corpus, desc='Subsampling frequent words'):
        subsampled_words = []
        for token in doc.text:
            if token in discard:
                continue
            if token not in UNK_filtered_freq:
                token = '[UNK]'
            if random.random() < keep_prob[token]:
                subsampled_words.append(token)

        for faux_sent in faux_sent_tokenize(
                subsampled_words, fixed_sent_len, min_sent_len):
            final_freq.update(faux_sent)
            doc.sentences.append(Sentence(subsampled_tokens=faux_sent))
            for word in faux_sent:
                if word not in ground:
                    ground[word] = GroundedWord(
                        text=word, deno=None, cono=Counter({doc.party: 1}))
                else:
                    ground[word].cono[doc.party] += 1

        if conserve_RAM:
            doc.text = None
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

    # Prepare grounding for intrinsic evaluation
    random_eval_words = set()
    for gw in ground.values():
        gw.majority_cono = gw.cono.most_common(1)[0][0]
        gw.freq = sum(gw.cono.values())
        gw.R_ratio = gw.cono['R'] / gw.freq
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

    # Helper for negative sampling
    cumulative_freq = sum(freq ** 0.75 for freq in final_freq.values())
    negative_sampling_probs: Dict[int, float] = {
        word_to_id[word]: (freq ** 0.75) / cumulative_freq
        for word, freq in final_freq.items()}
    vocab_size = len(word_to_id)
    negative_sampling_probs: List[float] = [
        # negative_sampling_probs[word_id]  # strict
        negative_sampling_probs.get(word_id, 0)  # prob = 0 if missing vocab
        for word_id in range(vocab_size)]

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
            # print(sent.tokens, file=preview)
            # print(sent.normalized_tokens, file=preview)
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

    # preview_path = out_dir / 'preview.txt'
    # with open(preview_path, 'w') as preview:
    #     preview.write(f'final vocabulary size = {len(word_to_id):,}\n')
    #     # preview.write(f'total number of words = {num_words_exported:,}\n')
    #     preview.write(f'subsampling implementation = {subsampling_implementation}\n')
    #     preview.write(f'subsampling threshold = {subsampling_threshold}\n')
    #     preview.write(f'minimum word frequency = {min_word_freq}\n')
    #     preview.write(f'\nPreview:\n')
    #     for speech in export_docs[:100]:
    #         words = map(id_to_word.get, speech[1])  # type: ignore
    #         preview.write(' '.join(words) + '\n')

    # export_sorted_frequency(
    #     raw_frequency, combined_frequency, min_word_freq,
    #     out_dir / 'vocabulary_subsampled_frequency.txt')
    # export_sampled_frequency_by_party(
    #     D_raw_freq, R_raw_freq, D_final_freq, R_final_freq, word_to_id,
    #     out_dir / 'vocabulary_partisan_frequency')

    # random.shuffle(export_docs)
    # corpus = [doc for _, doc in export_docs]
    # labels = [label for label, _ in export_docs]

    # grounding: Dict[str, Dict] = {}
    # for word in word_to_id.keys():
    #     df = D_raw_freq[word]
    #     rf = R_raw_freq[word]
    #     grounding[word] = {
    #         'D': df,
    #         'R': rf,
    #         'R_ratio': rf / (df + rf)}

    # cucumbers = {
    #     'word_to_id': word_to_id,
    #     'id_to_word': id_to_word,
    #     'grounding': grounding,
    #     'negative_sampling_probs': negative_sampling_probs,
    #     'documents': documents,
    #     'cono_labels': labels}
    # train_data_path = out_dir / f'train_data.pickle'
    # with open(train_data_path, 'wb') as export_file:
    #     pickle.dump(cucumbers, export_file, protocol=-1)
    # print('All set.')


if __name__ == '__main__':
    main(
        sessions=range(97, 112),
        in_dir=Path('../../data/interim/underscored_corpora'),
        out_dir=Path(f'../../data/ready/CR_proxy'),
        subsampling_implementation='paper',
        subsampling_threshold=1e-5,
        min_word_freq=15,
        min_sent_len=5,
        fixed_sent_len=20,
        eval_min_freq=100,
        eval_R_thresholds=(0.6, 0.7, 0.75, 0.8, 0.9),
        eval_num_random_samples=500,
        conserve_RAM=True
    )
