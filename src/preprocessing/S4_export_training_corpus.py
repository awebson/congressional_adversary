import pickle
import math
import random
import os
from collections import Counter, defaultdict
from typing import Tuple, NamedTuple, List, Dict, Iterable, Optional
from typing import Counter as CounterType

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

random.seed(1)
# Sentence = namedtuple('Sentence', ['word_ids', 'party'])
punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'  # excluding underscore
remove_punctutation = str.maketrans('', '', punctuations)
remove_numbers = str.maketrans('', '', '0123456789')


class Sentence(NamedTuple):
    word_ids: List[int]
    party: int  # Dem -> 0; GOP -> 1.


def subsampling(
        frequency: CounterType[str],
        heuristic: str,
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


def sort_freq_and_write(
        raw_frequency: CounterType[str],
        subsampled_frequency: CounterType[str],
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


# def negative_sampling(
#         subsampled_frequency: CounterType[str],
#         id_to_word: Dict[int, str]
#         ) -> List[float]:
#     """
#     A smoothed unigram distribution.
#     A simplified case of Noise Contrasitive Estimate.
#     where the seemingly aribitrary number of 0.75 is from the paper.
#     """
#     cumulative_freq = sum(freq ** 0.75 for freq in subsampled_frequency.values())
#     sampling_dist: Dict[str, float] = {
#         word: (freq ** 0.75) / cumulative_freq
#         for word, freq in subsampled_frequency.items()
#     }
#     multinomial_dist_probs = [
#         sampling_dist.get(id_to_word[word_id], 0)
#         for word_id in range(len(id_to_word))
#     ]
#     return multinomial_dist_probs


def export_plain_text(
        sessions: Iterable[int],
        output_dir: str,
        input_dir: str) -> None:
    output_path = os.path.join(output_dir, 'corpus.txt')
    output_file = open(output_path, 'w')

    num_words_exported = 0
    for session in tqdm(sessions, desc='Loading underscored corpora'):
        for party in ('D', 'R'):
            underscored_path = os.path.join(
                input_dir, f'underscored_{party}{session}.txt')
            with open(underscored_path, 'r') as underscored_corpus:
                for line in underscored_corpus:
                    num_words_exported += len(line.split())
                    # For debugging a subset of corpus
                    # if num_words_exported > 1_000_000:
                    #     output_file.close()
                    #     return
                    output_file.write(line)
    output_file.close()
    print(f'Total number of words = {num_words_exported:,}')


def export_speech_for_labeling(
        sessions: Iterable[int],
        output_dir: str,
        input_dir: str) -> None:
    output_path = os.path.join(output_dir, 'corpus.txt')
    output_file = open(output_path, 'w')

    num_words_exported = 0
    for session in tqdm(sessions, desc='Loading underscored corpora'):
        for party in ('D', 'R'):
            underscored_path = os.path.join(
                input_dir, f'underscored_{party}{session}.txt')
            with open(underscored_path, 'r') as underscored_corpus:
                for line in underscored_corpus:
                    num_words_exported += len(line.split())
                    # For debugging a subset of corpus
                    # if num_words_exported > 1_000_000:
                    #     output_file.close()
                    #     return
                    output_file.write(line)
    output_file.close()
    print(f'Total number of words = {num_words_exported:,}')


def build_vocabulary(
        frequency: Counter,
        min_frequency: int
        ) -> Tuple[
        Dict[str, int],
        Dict[int, str]]:
    word_to_id: Dict[str, int] = {}
    id_to_word: Dict[int, str] = {}
    next_vocab_id = 0
    for word, freq in frequency.items():
        if word not in word_to_id and freq > min_frequency:
            word_to_id[word] = next_vocab_id
            id_to_word[next_vocab_id] = word
            next_vocab_id += 1
    print(f'Vocabulary size = {len(word_to_id):,}')
    return word_to_id, id_to_word


def faux_sent_tokenize(words: List[str], fixed_sent_len: int) -> Iterable[List[str]]:
    # words = line.translate(remove_punctutation).split()
    start_index = 0
    while (start_index + fixed_sent_len) < (len(words) - 1):
        yield words[start_index:start_index + fixed_sent_len]
        start_index += fixed_sent_len
    yield words[start_index:-1]


def balance_classes(
        Dem_speeches: List[Sentence],
        GOP_speeches: List[Sentence]
        ) -> List[Sentence]:
    total = len(Dem_speeches) + len(GOP_speeches)
    print(f'Pre-balanced Dem speech ratio = {len(Dem_speeches) / total:.4%}')
    print(f'Pre-balanced GOP speech ratio = {len(GOP_speeches) / total:.4%}')
    minority = min(len(GOP_speeches), len(Dem_speeches))
    if len(GOP_speeches) > len(Dem_speeches):
        GOP_speeches = random.sample(GOP_speeches, k=minority)
        print(f'Balancing training data by sampling GOP to {minority:,}.')
    else:
        Dem_speeches = random.sample(Dem_speeches, k=minority)
        print(f'Balancing training data by sampling Dem to {minority:,}.')
    return Dem_speeches + GOP_speeches


def export_party_classification_data(
        sessions: Iterable[int],
        output_dir: str,
        input_dir: str,
        subsampling_implementation: Optional[str],
        subsampling_threshold: float,
        min_word_freq: int,
        min_sentence_len: int
        ) -> None:
    """
    Sentence tokenized, with metadata per sentence.
    # TODO multiprcessing

    Choose subsampling_implementaiton among 'paper', 'code', or None.
    """
    raw_frequency: CounterType[str] = Counter()
    for session in tqdm(sessions, desc='Building vocabulary from underscored corpora'):
        for party in ('D', 'R'):
            underscored_path = os.path.join(
                input_dir, f'underscored_{party}{session}.txt')
            with open(underscored_path, 'r') as underscored_corpus:
                for speech in underscored_corpus:
                    speech = speech.translate(remove_punctutation)
                    words = speech.split()
                    raw_frequency.update(words)
    word_to_id, id_to_word = build_vocabulary(raw_frequency, min_word_freq)

    if subsampling_implementation:
        keep_prob = subsampling(
            raw_frequency, subsampling_implementation, subsampling_threshold)
    else:
        keep_prob = defaultdict(lambda: 1)

    # num_words_exported = 0
    Dem_speeches: List[Sentence] = []
    GOP_speeches: List[Sentence] = []
    subsampled_frequency: CounterType[str] = Counter()
    for session in tqdm(sessions, desc=' Exporting Sessions'):
        for party in ('D', 'R'):
            underscored_path = os.path.join(
                input_dir, f'underscored_{party}{session}.txt')
            with open(underscored_path, 'r') as underscored_corpus:
                for speech in underscored_corpus:
                    # Export full speech.
                    # sentences = [speech, ]
                    # for sent in sentences:
                    #     sent = sent.translate(remove_punctutation)
                    #     subsampled_words = [
                    #         word for word in sent.split()
                    #         if (raw_frequency[word] > min_word_freq
                    #             and random.random() < keep_prob[word])
                    #     ]
                    #     if len(subsampled_words) < min_sentence_len:
                    #         continue
                    #     num_words_exported += len(subsampled_words)
                    #     subsampled_frequency.update(subsampled_words)
                    #     subsampled_word_ids = [
                    #         word_to_id[word] for word in subsampled_words]
                    #     if party == 'D':
                    #         num_Dem_speeches += 1
                    #         export_sentences.append(
                    #             Sentence(subsampled_word_ids, 0))
                    #     else:
                    #         num_GOP_speeches += 1
                    #         export_sentences.append(
                    #             Sentence(subsampled_word_ids, 1))

                    # Tokenize sentences with fixed length.
                    speech = speech.translate(remove_punctutation)
                    subsampled_speech = [
                        word for word in speech.split()
                        if (raw_frequency[word] > min_word_freq
                            and random.random() < keep_prob[word])
                    ]
                    for subsampled_words in faux_sent_tokenize(subsampled_speech, 30):
                        if len(subsampled_words) < min_sentence_len:
                            continue
                        # num_words_exported += len(subsampled_words)
                        subsampled_frequency.update(subsampled_words)
                        subsampled_word_ids = [
                            word_to_id[word] for word in subsampled_words]
                        if party == 'D':
                            Dem_speeches.append(
                                Sentence(subsampled_word_ids, 0))
                        else:
                            GOP_speeches.append(
                                Sentence(subsampled_word_ids, 1))

    export_speeches = balance_classes(Dem_speeches, GOP_speeches)  # TODO subsample frequency changed again
    num_speeches = len(export_speeches)
    print(f'Total number of speeches = {num_speeches:,}')

    # random.shuffle(export_sentences)
    # word_to_id, id_to_word = build_vocabulary(subsampled_frequency, min_frequency=0)  # TODO necessary?

    preview_path = os.path.join(output_dir, 'preview.txt')
    with open(preview_path, 'w') as preview:
        preview.write(f'vocabulary size = {len(word_to_id):,}\n')
        # preview.write(f'total number of words = {num_words_exported:,}\n')
        preview.write(f'total number of speeches = {num_speeches:,}\n')
        # preview.write(f'Pre-balanced Dem speech ratio = {num_Dem_speeches / num_speeches:.4%}\n')
        # preview.write(f'Pre-balanced GOP speech ratio = {num_GOP_speeches / num_speeches:.4%}\n')
        preview.write(f'subsampling implementation = {subsampling_implementation}\n')
        preview.write(f'subsampling threshold = {subsampling_threshold}\n')
        preview.write(f'minimum word frequency = {min_word_freq}\n')
        preview.write(f'\nPreview:\n')
        for sentence in export_speeches:
            words = map(id_to_word.get, sentence.word_ids)  # type: ignore
            if sentence.party == 0:
                preview.write(f'D: {" ".join(words)}\n')
            else:
                preview.write(f'R: {" ".join(words)}\n')

    sort_freq_and_write(
        raw_frequency, subsampled_frequency, min_word_freq,
        os.path.join(output_dir, 'vocabulary_frequency.txt'))

    train_data_path = os.path.join(output_dir, f'train_data.pickle')
    with open(train_data_path, 'wb') as export_file:
        pickle.dump(
            (word_to_id, id_to_word, subsampled_frequency, export_speeches),
            export_file, protocol=-1)
    print('All set.')


def export_presidential_party_classification(
        input_path: str,
        output_dir: str,
        subsampling_implementation: Optional[str],
        subsampling_threshold: float,
        min_word_freq: int,
        min_sentence_len: int
        ) -> None:
    import json
    polisci101 = {
        'Donald J. Trump': 1,
        'Barack Obama': 0,
        'George W. Bush': 1,
        'William J. Clinton': 0,
        'George Bush': 1,
        'Ronald Reagan': 1,
        'Jimmy Carter': 0,
        'Gerald R. Ford': 1,
        'Richard Nixon': 1,
        'Lyndon B. Johnson': 0,
        'John F. Kennedy': 0,
        'Dwight D. Eisenhower': 1,
        'Harry S. Truman': 0
    }
    speech_json: List[Dict] = []
    raw_frequency: CounterType[str] = Counter()
    skipped: CounterType[str] = Counter()
    with open(input_path, 'r') as jsonl_file:
        for line in jsonl_file:
            json_obj = json.loads(line)
            name = json_obj['person']
            if name not in polisci101:
                skipped[name] += 1
                continue
            try:
                raw = ''.join(json_obj['speech'])
            except KeyError:
                continue
            words = (raw.lower()
                     .translate(remove_punctutation)
                     .translate(remove_numbers)
                     .split())
            raw_frequency.update(words)
            json_obj['speech'] = words
            json_obj['party'] = polisci101[name]
            speech_json.append(json_obj)
    print('Skipped the following undefined people: ', skipped)
    word_to_id, id_to_word = build_vocabulary(raw_frequency, min_word_freq)

    if subsampling_implementation:
        keep_prob = subsampling(
            raw_frequency, subsampling_implementation, subsampling_threshold)
    else:
        keep_prob = defaultdict(lambda: 1)

    Dem_speeches: List[Sentence] = []
    GOP_speeches: List[Sentence] = []
    subsampled_frequency: CounterType[str] = Counter()
    for json_obj in speech_json:
        subsampled_speech = [
            word for word in json_obj['speech']
            if (raw_frequency[word] > min_word_freq
                and random.random() < keep_prob[word])
        ]
        # Tokenize sentences with fixed length.
        for subsampled_words in faux_sent_tokenize(subsampled_speech, 30):
            if len(subsampled_words) < min_sentence_len:
                continue
            # num_words_exported += len(subsampled_words)
            subsampled_frequency.update(subsampled_words)
            subsampled_word_ids = [
                word_to_id[word] for word in subsampled_words]
            if json_obj['party'] == 0:
                Dem_speeches.append(
                    Sentence(subsampled_word_ids, 0))
            else:
                GOP_speeches.append(
                    Sentence(subsampled_word_ids, 1))

    export_speeches = balance_classes(Dem_speeches, GOP_speeches)  # TODO subsample frequency changed again
    num_speeches = len(export_speeches)
    print(f'Total number of speeches = {num_speeches:,}')

    # random.shuffle(export_sentences)
    # word_to_id, id_to_word = build_vocabulary(subsampled_frequency, min_frequency=0)  # TODO necessary?

    preview_path = os.path.join(output_dir, 'preview.txt')
    with open(preview_path, 'w') as preview:
        preview.write(f'vocabulary size = {len(word_to_id):,}\n')
        # preview.write(f'total number of words = {num_words_exported:,}\n')
        preview.write(f'total number of speeches = {num_speeches:,}\n')
        # preview.write(f'Pre-balanced Dem speech ratio = {num_Dem_speeches / num_speeches:.4%}\n')
        # preview.write(f'Pre-balanced GOP speech ratio = {num_GOP_speeches / num_speeches:.4%}\n')
        preview.write(f'subsampling implementation = {subsampling_implementation}\n')
        preview.write(f'subsampling threshold = {subsampling_threshold}\n')
        preview.write(f'minimum word frequency = {min_word_freq}\n')
        preview.write(f'\nPreview:\n')
        for sentence in export_speeches:
            words = map(id_to_word.get, sentence.word_ids)  # type: ignore
            if sentence.party == 0:
                preview.write(f'D: {" ".join(words)}\n')
            else:
                preview.write(f'R: {" ".join(words)}\n')

    sort_freq_and_write(
        raw_frequency, subsampled_frequency, min_word_freq,
        os.path.join(output_dir, 'vocabulary_frequency.txt'))

    train_data_path = os.path.join(output_dir, f'train_data.pickle')
    with open(train_data_path, 'wb') as export_file:
        pickle.dump(
            (word_to_id, id_to_word, subsampled_frequency, export_speeches),
            export_file, protocol=-1)
    print('All set.')


def export_skip_gram_data(
        sessions: Iterable[int],
        output_dir: str,
        input_dir: str,
        # output_num_chunks: int,
        subsampling_implementation: Optional[str],
        subsampling_threshold: float,
        min_word_freq: int
        ) -> None:
    """
    Shuffling requires inefficently loading all corpora in memory.
    """
    raw_frequency: CounterType[str] = Counter()
    tokenized_speeches: List[List[str]] = []
    for session in tqdm(sessions, desc='Loading underscored corpora'):
        for party in ('D', 'R'):
            underscored_path = os.path.join(
                input_dir, f'underscored_{party}{session}.txt')
            with open(underscored_path, 'r') as underscored_corpus:
                for line in underscored_corpus:
                    tokenized_speech = line.split()
                    raw_frequency.update(tokenized_speech)
                    tokenized_speeches.append(tokenized_speech)

    random.shuffle(tokenized_speeches)
    word_to_id, id_to_word = build_vocabulary(raw_frequency, min_word_freq)
    if subsampling_implementation:
        keep_prob = subsampling(
            raw_frequency, subsampling_implementation, subsampling_threshold)
    else:
        keep_prob = defaultdict(lambda: 1)

    num_words_exported = 0
    subsampled_frequency: CounterType[str] = Counter()

    # Chunked Implementation
    # for chunk_index, speeches_chunk in tqdm(enumerate(
    #         partition(tokenized_speeches, output_num_chunks)),
    #         desc='Exporting Chunks', total=output_num_chunks):

    #     export_speeches: List[List[int]] = []
    #     for tokenized_speech in speeches_chunk:

    export_speeches: List[List[int]] = []
    for tokenized_speech in tqdm(tokenized_speeches, desc='Exporting speeches'):
        export_speech = [
            word for word in tokenized_speech
            if (raw_frequency[word] > min_word_freq
                and random.random() < keep_prob[word])
        ]
        if len(export_speech) < 2:
            continue
        num_words_exported += len(export_speech)
        subsampled_frequency.update(export_speech)
        export_speech_in_word_ids = [word_to_id[word] for word in export_speech]
        export_speeches.append(export_speech_in_word_ids)

    train_data_path = os.path.join(output_dir, f'train_data_0.pickle')
    with open(train_data_path, 'wb') as export_file:
        pickle.dump(export_speeches, export_file, protocol=-1)

    print(f'Total number of words = {num_words_exported:,}')
    preview_path = os.path.join(output_dir, 'preview.txt')
    with open(preview_path, 'w') as preview_file:
        preview_file.write(f'vocabulary size = {len(word_to_id):,}\n')
        preview_file.write(f'total number of words = {num_words_exported:,}\n')
        preview_file.write(f'subsampling method = {subsampling_implementation}\n')
        preview_file.write(f'subsampling threshold = {subsampling_threshold}\n')
        preview_file.write(f'minimum word frequency = {min_word_freq}\n')
        preview_file.write(f'\nPreview:\n')
        for speech_in_word_ids in export_speeches[:100]:
            speech = map(id_to_word.get, speech_in_word_ids)
            preview_file.write(' '.join(speech) + '\n')  # type: ignore

    sort_freq_and_write(
        raw_frequency, subsampled_frequency, min_word_freq,
        os.path.join(output_dir, 'vocabulary_frequency.txt'))

    negative_sampling_dist = negative_sampling(subsampled_frequency, id_to_word)
    vocab_path = os.path.join(output_dir, 'vocab.pickle')
    with open(vocab_path, 'wb') as vocab_file:
        pickle.dump((word_to_id, id_to_word, negative_sampling_dist),
                    vocab_file, protocol=-1)

    # sanity check negative sampling
    # for word_id, word in id_to_word.items():
    #     print(f'{word}\t{subsampled_frequency[word]:,}\t{negative_sampling_dist[word_id]:.2e}')
    print('All set.')


def end_to_end(
        corpus_path: str,
        output_dir: str,
        output_num_chunks: int,
        subsampling_implementation: Optional[str],
        subsampling_threshold: float,
        min_word_freq: int
        ) -> None:
    """
    Similar to export_skip_gram_negative_sampling, but procesing a raw text corpus.
    """
    discard_char = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~_0123456789'
    remove_punctutations_and_numbers = str.maketrans('', '', discard_char)

    raw_frequency: CounterType[str] = Counter()
    tokenized_corpus: List[List[str]] = []
    with open(corpus_path, 'r') as corpus_file:
        for line in corpus_file:
            line = line.translate(remove_punctutations_and_numbers)
            tokens = line.split()
            raw_frequency.update(tokens)
            tokenized_corpus.append(tokens)

    random.shuffle(tokenized_corpus)
    sanity_check_path = os.path.join(output_dir, 'sanity_check.txt')
    with open(sanity_check_path, 'w') as sanity_check_file:
        for index, tokens in enumerate(tokenized_corpus):
            for word in tokens:
                sanity_check_file.write(word + ' ')
            sanity_check_file.write('\n')
            if index > 100:
                break

    word_to_id: Dict[str, int] = {}
    id_to_word: Dict[int, str] = {}
    next_vocab_id = 0
    for word, freq in raw_frequency.items():
        if word not in word_to_id and freq > min_word_freq:
            word_to_id[word] = next_vocab_id
            id_to_word[next_vocab_id] = word
            next_vocab_id += 1
    print(f'Vocabulary size = {len(word_to_id):,}')

    if subsampling_implementation:
        keep_prob = subsampling(
            raw_frequency, subsampling_implementation, subsampling_threshold)
    else:
        keep_prob = defaultdict(lambda: 1)

    num_words_exported = 0
    subsampled_frequency: CounterType[str] = Counter()
    for chunk_index, speeches_chunk in tqdm(enumerate(
            partition(tokenized_corpus, output_num_chunks)),
            desc='Exporting Chunks', total=output_num_chunks):

        export_speeches: List[List[int]] = []
        for tokenized_speech in speeches_chunk:
            export_speech = [
                word for word in tokenized_speech
                if (raw_frequency[word] > min_word_freq
                    and random.random() < keep_prob[word])
            ]
            num_words_exported += len(export_speech)
            subsampled_frequency.update(export_speech)
            export_speech_in_word_ids = [word_to_id[word] for word in export_speech]
            export_speeches.append(export_speech_in_word_ids)

        train_data_path = os.path.join(output_dir, f'train_data_{chunk_index}.pickle')
        with open(train_data_path, 'wb') as export_file:
            pickle.dump(export_speeches, export_file, protocol=-1)
    print(f'Total number of words = {num_words_exported:,}')

    sort_freq_and_write(
        raw_frequency, subsampled_frequency, min_word_freq,
        os.path.join(output_dir, 'vocabulary_frequency.txt'))

    negative_sampling_dist = negative_sampling(subsampled_frequency, id_to_word)
    vocab_path = os.path.join(output_dir, 'vocab.pickle')
    with open(vocab_path, 'wb') as vocab_file:
        pickle.dump((word_to_id, id_to_word, negative_sampling_dist),
                    vocab_file, protocol=-1)
    print('All set.')


def main() -> None:
    # Just the corpus in plain text, no word_id. For external libraries
    # sessions = range(102, 112)
    # output_dir = '../corpora/skip_gram/plain_decade'
    # underscored_dir = 'underscored_corpora'
    # os.makedirs(output_dir, exist_ok=True)
    # export_plain_text(sessions, output_dir, underscored_dir)

    # # Shuffled corpus with each word replaced by its unique word_id
    # # sessions = range(111, 112)  # just one session for debugging
    # sessions = range(102, 112)  # 2000s
    # # sessions = range(79, 112)   # post-WWII
    # output_dir = '../../data/processed/party_classifier/2000s_len30sent_1e-3'
    # subsampling_implementation: Optional[str] = 'paper'
    # subsampling_threshold = 1e-3
    # # output_num_chunks = 1
    # underscored_dir = '../../data/interim/underscored_corpora'
    # min_word_freq = 15
    # min_sentence_len = 5
    # os.makedirs(output_dir, exist_ok=True)
    # print(f'Reading {sessions}. Writing to {output_dir}')
    # export_party_classification_data(
    #     sessions, output_dir, underscored_dir,  # output_num_chunks,
    #     subsampling_implementation, subsampling_threshold,
    #     min_word_freq, min_sentence_len)

    # Party classification from JSONL, no underscore phrases
    input_path = '../../data/raw/UCSB_presidency_project.jsonl'
    output_dir = '../../data/processed/party_classifier/UCSB_1e-3'
    subsampling_implementation: Optional[str] = 'paper'
    subsampling_threshold = 1e-3
    min_word_freq = 15
    min_sentence_len = 5
    os.makedirs(output_dir, exist_ok=True)
    print(f'Reading {input_path}. Writing to {output_dir}')
    export_presidential_party_classification(
        input_path, output_dir,  # output_num_chunks,
        subsampling_implementation, subsampling_threshold,
        min_word_freq, min_sentence_len)

    # skip-gram negative sampling from raw text, no underscore phrases
    # corpus_path = '~/Research/common_corpora/wikipedia_min20pageviews.txt'
    # output_dir = '../corpora/skip_gram/no_underscore'
    # corpus_path = os.path.expanduser(corpus_path)
    # output_num_chunks = 1
    # subsampling_implementation: Optional[str] = 'code'
    # subsampling_threshold = 1e-5
    # min_word_freq = 15
    # end_to_end(corpus_path, output_dir, output_num_chunks, subsampling_implementation, subsampling_threshold, min_word_freq)

    # Party Prediction
    # sessions = range(79, 112)
    # output_dir = '../corpora/party_classification/1e-5_10chunks'
    # output_num_chunks = 10
    # os.makedirs(output_dir, exist_ok=True)
    # underscored_dir = 'underscored_corpora'
    # subsampling_implementation: Optional[str] = 'code'
    # subsampling_threshold = 1e-5
    # test_data_per_session = 100
    # min_word_freq = 15
    # os.makedirs(output_dir, exist_ok=True)
    # print(f'Reading {sessions}. Writing to {output_dir}')
    # export_party_classification_data(
    #     sessions, output_dir, underscored_dir, output_num_chunks,
    #     subsampling_implementation, subsampling_threshold,
    #     min_word_freq, test_data_per_session)


if __name__ == '__main__':
    main()
