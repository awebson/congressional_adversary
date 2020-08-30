import pickle
import random
import math
import os
from collections import defaultdict
from typing import Tuple, List, Dict, Counter, Iterable, Optional

from tqdm import tqdm

from data import GroundedWord, Sentence, LabeledDoc


random.seed(1)

punctuations = '!"#$%&\'()*+,-—./:;<=>?@[\\]^`{|}~'  # excluding underscore
remove_punctutation = str.maketrans('', '', punctuations)
remove_numbers = str.maketrans('', '', '0123456789')
# party_to_label = {
#     'D': 0,
#     'R': 1
# }

# class Document(NamedTuple):
#     word_ids: List[int]
#     party: int


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
        keep_prob = defaultdict(lambda: 1)
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


def export_plain_text_corpus(
        sessions: Iterable[int],
        output_dir: str,
        input_dir: str
        ) -> None:
    output_file = open(os.path.join(output_dir, 'corpus.txt'), 'w')
    num_words_exported = 0
    for session in tqdm(sessions, desc='Loading underscored corpora'):
        for party in ('D', 'R'):
            underscored_path = os.path.join(
                input_dir, f'underscored_{party}{session}.txt')
            with open(underscored_path, 'r') as underscored_corpus:
                for line in underscored_corpus:
                    num_words_exported += len(line.split())
                    output_file.write(line)
    output_file.close()
    print(f'Total number of words = {num_words_exported:,}')


def count_partisan_frequency(
        sessions: Iterable[int],
        input_dir: str
        ) -> Tuple[Counter[str], Counter[str], Counter[str]]:
    D_freq: Counter[str] = Counter()
    R_freq: Counter[str] = Counter()
    for session in tqdm(sessions, desc='Counting partisan frequency'):
        for party in ('D', 'R'):
            underscored_path = os.path.join(
                input_dir, f'underscored_{party}{session}.txt')
            with open(underscored_path, 'r') as underscored_corpus:
                for line in underscored_corpus:
                    # line = line.translate(remove_punctutation)  # Done in S3
                    words = line.split()
                    if party == 'D':
                        D_freq.update(words)
                    else:
                        R_freq.update(words)
    combined_frequency = D_freq + R_freq
    return combined_frequency, D_freq, R_freq


def build_vocabulary(
        frequency: Counter,
        min_frequency: int,
        add_special_tokens: bool
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
        input_dir: str,
        training_min_freq: int,
        sort_min_freq: int
        ) -> None:
    D_freq: Counter[str] = Counter()
    R_freq: Counter[str] = Counter()
    for session in tqdm(sessions, desc='Loading underscored corpora'):
        for party in ('D', 'R'):
            underscored_path = os.path.join(
                input_dir, f'underscored_{party}{session}.txt')
            with open(underscored_path, 'r') as underscored_corpus:
                for line in underscored_corpus:
                    words = line.split()
                    if party == 'D':
                        D_freq.update(words)
                    else:
                        R_freq.update(words)
    word_to_id, _ = build_vocabulary(D_freq + R_freq, training_min_freq)
    output_path = os.path.join(output_dir, 'vocab_partisan_frequency.tsv')
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


def export_labeled_documents(
        sessions: Iterable[int],
        output_dir: str,
        input_dir: str,
        subsampling_implementation: Optional[str],
        subsampling_threshold: float,
        min_word_freq: int,
        min_document_len: int
        ) -> None:
    from nltk.corpus import stopwords
    procedural_words = {
        'yield', 'motion', 'order', 'ordered', 'quorum', 'roll', 'unanimous',
        'mr', 'madam', 'speaker', 'chairman', 'president', 'senator',
        'gentleman', 'colleague',
        'today', 'rise', 'rise today', 'pleased_to_introduce',
        'introducing_today', 'would_like'
    }
    stop_words = set(stopwords.words('english')).union(procedural_words)

    raw_frequency, D_raw_freq, R_raw_freq = count_partisan_frequency(
        sessions, input_dir)
    word_to_id, id_to_word = build_vocabulary(raw_frequency, min_word_freq)
    keep_prob = subsampling(
        raw_frequency, subsampling_implementation, subsampling_threshold)

    Document = Tuple[int, List[int]]  # tuple(party_label, word_ids)
    Dem_speeches: List[Document] = []
    GOP_speeches: List[Document] = []
    D_freq: Counter[str] = Counter()
    R_freq: Counter[str] = Counter()
    for session in tqdm(sessions, desc='Processing Sessions'):
        for party in ('D', 'R'):
            underscored_path = os.path.join(
                input_dir, f'underscored_{party}{session}.txt')
            with open(underscored_path, 'r') as underscored_corpus:
                for line in underscored_corpus:
                    # line = line.translate(remove_punctutation)  # Done in S3
                    words = line.split()
                    subsampled_word_ids = [
                        word_to_id[word] for word in words
                        if (word not in stop_words
                            and raw_frequency[word] > min_word_freq
                            and random.random() < keep_prob[word])
                    ]
                    if len(subsampled_word_ids) < min_document_len:
                        continue
                    if party == 'D':
                        D_freq.update(words)
                        Dem_speeches.append((0, subsampled_word_ids))
                    else:
                        R_freq.update(words)
                        GOP_speeches.append((1, subsampled_word_ids))
    export_docs = balance_classes(Dem_speeches, GOP_speeches)
    # export_docs = Dem_speeches + GOP_speeches  # no balancing

    Dem_word_count = 0
    GOP_word_count = 0
    D_final_freq: Counter[str] = Counter()
    R_final_freq: Counter[str] = Counter()
    combined_frequency: Counter[str] = Counter()
    for doc in export_docs:
        # doc: tuple(party_label, word_ids)
        words = [id_to_word[word_id] for word_id in doc[1]]
        combined_frequency.update(words)
        if doc[0] == 0:
            D_final_freq.update(words)
            Dem_word_count += len(doc[1])
        else:
            R_final_freq.update(words)
            GOP_word_count += len(doc[1])
    print(f'Pre-sampled vocab size = {len(word_to_id):,}')
    print(f'Post-sampled vocab size = {len(combined_frequency):,}')
    print(f'Post-balanced Dem word count = {Dem_word_count:,}')
    print(f'Post-balanced GOP word count = {GOP_word_count:,}')

    # Optionally rebuild vocabulary after sampling
    # word_to_id, id_to_word = build_vocabulary(
    #     combined_frequency, min_frequency=0)

    cumulative_freq = sum(freq ** 0.75 for freq in combined_frequency.values())
    negative_sampling_probs: Dict[int, float] = {
        word_to_id[word]: (freq ** 0.75) / cumulative_freq
        for word, freq in combined_frequency.items()
    }
    vocab_size = len(word_to_id)
    negative_sampling_probs: List[float] = [
        # negative_sampling_probs[word_id]  # strict
        negative_sampling_probs.get(word_id, 0)  # prob = 0 if missing vocab
        for word_id in range(vocab_size)
    ]
    preview_path = os.path.join(output_dir, 'preview.txt')
    with open(preview_path, 'w') as preview:
        preview.write(f'final vocabulary size = {len(word_to_id):,}\n')
        # preview.write(f'total number of words = {num_words_exported:,}\n')
        preview.write(f'subsampling implementation = {subsampling_implementation}\n')
        preview.write(f'subsampling threshold = {subsampling_threshold}\n')
        preview.write(f'minimum word frequency = {min_word_freq}\n')
        preview.write(f'\nPreview:\n')
        for speech in export_docs[:100]:
            words = map(id_to_word.get, speech[1])  # type: ignore
            preview.write(' '.join(words) + '\n')

    export_sorted_frequency(
        raw_frequency, combined_frequency, min_word_freq,
        os.path.join(output_dir, 'vocabulary_subsampled_frequency.txt'))
    export_sampled_frequency_by_party(
        D_raw_freq, R_raw_freq, D_final_freq, R_final_freq, word_to_id,
        os.path.join(output_dir, 'vocabulary_partisan_frequency'))

    random.shuffle(export_docs)
    documents = [doc for _, doc in export_docs]
    labels = [label for label, _ in export_docs]

    grounding: Dict[str, Dict] = {}
    for word in word_to_id.keys():
        df = D_raw_freq[word]
        rf = R_raw_freq[word]
        grounding[word] = {
            'D': df,
            'R': rf,
            'R_ratio': rf / (df + rf)}

    cucumbers = {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'grounding': grounding,
        'negative_sampling_probs': negative_sampling_probs,
        'documents': documents,
        'cono_labels': labels}
    train_data_path = os.path.join(output_dir, f'train_data.pickle')
    with open(train_data_path, 'wb') as export_file:
        pickle.dump(cucumbers, export_file, protocol=-1)
    print('All set.')


def main() -> None:
    # during: Dict[str, Iterable[int]] = {
    #     'postwar': range(79, 112),
    #     'Truman': range(79, 83),
    #     'Eisenhower': range(83, 87),
    #     'Kennedy': range(87, 88),
    #     'Johnson': range(88, 91),
    #     'Nixon': range(91, 94),
    #     'Ford': range(94, 95),
    #     'Carter': range(95, 97),
    #     'Reagan': range(97, 101),
    #     'H.W.Bush': range(101, 103),
    #     'Clinton': range(103, 107),
    #     'W.Bush': range(107, 111),
    #     'Obama': range(111, 112)  # incomplete data in the Bound edition
    # }

    # # Just the corpus in plain text, no word_id. For external libraries
    # sessions = range(97, 112)  # during['W.Bush']
    # output_dir = '../../data/processed/plain_text/for_real'
    # underscored_dir = '../../data/interim/underscored_corpora'
    # os.makedirs(output_dir, exist_ok=True)
    # # export_plain_text_corpus(sessions, output_dir, underscored_dir)

    # # For labeling evaluation data
    # export_sorted_frequency_by_party(
    #     sessions, output_dir, underscored_dir,
    #     training_min_freq=15, sort_min_freq=100)

    # # Skip-Gram Negative Sampling
    # sessions = during['Obama']
    # output_dir = '../../data/processed/skip_gram/Obama_1e-5'
    # subsampling_implementation: Optional[str] = 'paper'
    # subsampling_threshold = 1e-5
    # underscored_dir = '../../data/interim/underscored_corpora'
    # min_word_freq = 15
    # min_speech_length = 20
    # os.makedirs(output_dir, exist_ok=True)
    # print(f'Reading {sessions}. Writing to {output_dir}')
    # export_numericalized_corpus(
    #     sessions, output_dir, underscored_dir,  # output_num_chunks,
    #     subsampling_implementation, subsampling_threshold,
    #     min_word_freq, min_speech_length)

    # # preparsed SGNS
    # sessions = during['Obama']
    # output_dir = '../../data/processed/skip_gram_preparsed/tuple_Obama_1e-5'
    # window_radius = 5
    # subsampling_implementation: Optional[str] = 'paper'
    # subsampling_threshold = 1e-5
    # underscored_dir = '../../data/interim/underscored_corpora'
    # min_word_freq = 15
    # min_speech_length = 20
    # os.makedirs(output_dir, exist_ok=True)
    # print(f'Reading {sessions}. Writing to {output_dir}')
    # export_skip_grams(
    #     sessions, output_dir, underscored_dir, window_radius,
    #     subsampling_implementation, subsampling_threshold,
    #     min_word_freq, min_speech_length)

    # SGNS from raw text, no underscore phrases
    # corpus_path = '~/Research/common_corpora/wikipedia_min20pageviews.txt'
    # output_dir = '../corpora/skip_gram/no_underscore'
    # corpus_path = os.path.expanduser(corpus_path)
    # output_num_chunks = 1
    # subsampling_implementation: Optional[str] = 'code'
    # subsampling_threshold = 1e-5
    # min_word_freq = 15
    # end_to_end(corpus_path, output_dir, output_num_chunks, subsampling_implementation, subsampling_threshold, min_word_freq)

    # # Word-level party classifier
    # sessions = range(97, 112)
    # underscored_dir = '../../data/interim/underscored_corpora'
    # output_dir = '../../data/processed/labeled_words/1e-5/for_real'
    # subsampling_implementation: Optional[str] = 'paper'
    # subsampling_threshold = 1e-5
    # min_word_freq = 15
    # min_document_len = 20
    # os.makedirs(output_dir, exist_ok=True)
    # print(f'Writing to {output_dir}')
    # export_labeled_words(
    #     sessions, output_dir, underscored_dir,
    #     subsampling_implementation, subsampling_threshold,
    #     min_word_freq, min_document_len)

    # # 5-Bucket Word-level party classifier
    # sessions = range(97, 112)
    # underscored_dir = '../../data/interim/underscored_corpora'
    # output_dir = '../../data/processed/bucket_labeled_words/1e-5/debug'
    # subsampling_implementation: Optional[str] = 'paper'
    # subsampling_threshold = 1e-5
    # min_word_freq = 15
    # min_document_len = 20
    # os.makedirs(output_dir, exist_ok=True)
    # print(f'Writing to {output_dir}')
    # export_bucket_labeled_words(
    #     sessions, output_dir, underscored_dir,
    #     subsampling_implementation, subsampling_threshold,
    #     min_word_freq, min_document_len)

    # # Adversarial
    # sessions = range(97, 112)
    # output_dir = f'../../data/processed/labeled_speeches/1e-5/Obama'
    # subsampling_implementation: Optional[str] = 'paper'
    # subsampling_threshold = 1e-5
    # underscored_dir = '../../data/interim/underscored_corpora'
    # min_word_freq = 15
    # min_speech_length = 20
    # os.makedirs(output_dir, exist_ok=True)
    # tqdm.write(f'Reading {sessions}. Writing to {output_dir}')
    # export_labeled_speeches(
    #     sessions, output_dir, underscored_dir,
    #     subsampling_implementation, subsampling_threshold,
    #     min_word_freq, min_speech_length)


    # # Speeches with 5-bucket id_to_label
    # sessions = range(97, 112)
    # output_dir = f'../../data/processed/bucket_labeled_speeches/1e-5/for_real'
    # subsampling_implementation: Optional[str] = 'paper'
    # subsampling_threshold = 1e-5
    # underscored_dir = '../../data/interim/underscored_corpora'
    # min_word_freq = 15
    # min_speech_length = 20
    # os.makedirs(output_dir, exist_ok=True)
    # tqdm.write(f'Reading {sessions}. Writing to {output_dir}')
    # export_bucket_labeled_speeches(
    #     sessions, output_dir, underscored_dir,
    #     subsampling_implementation, subsampling_threshold,
    #     min_word_freq, min_speech_length)


    # # Preparsed Adversarial
    # sessions = range(97, 112)
    # output_dir = f'../../data/processed/labeled_skip_grams/1e-5/for_real'
    # window_radius = 5
    # subsampling_implementation: Optional[str] = 'paper'
    # subsampling_threshold = 1e-5
    # underscored_dir = '../../data/interim/underscored_corpora'
    # min_word_freq = 15
    # min_speech_length = 20
    # os.makedirs(output_dir, exist_ok=True)
    # tqdm.write(f'Reading {sessions}. Writing to {output_dir}')
    # export_labeled_skip_grams(
    #     sessions, output_dir, underscored_dir, window_radius,
    #     subsampling_implementation, subsampling_threshold,
    #     min_word_freq, min_speech_length)

    sessions = range(97, 112)
    output_dir = f'../../data/processed/CR_skip'
    subsampling_implementation: Optional[str] = 'paper'
    subsampling_threshold = 1e-5
    underscored_dir = '../../data/interim/underscored_corpora'
    min_word_freq = 15
    min_speech_length = 20
    os.makedirs(output_dir, exist_ok=True)
    tqdm.write(f'Reading {sessions}. Writing to {output_dir}')
    export_labeled_documents(
        sessions, output_dir, underscored_dir,
        subsampling_implementation, subsampling_threshold,
        min_word_freq, min_speech_length)


if __name__ == '__main__':
    main()

    # import cProfile
    # cProfile.run('main()', sort='cumulative')
