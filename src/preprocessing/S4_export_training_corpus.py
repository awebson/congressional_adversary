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

punctuations = '!"#$%&\'()*+,-—./:;<=>?@[\\]^`{|}~'  # excluding underscore
remove_punctutation = str.maketrans('', '', punctuations)
remove_numbers = str.maketrans('', '', '0123456789')


class Document(NamedTuple):
    word_ids: List[int]
    party: int  # Dem -> 0; GOP -> 1.


def subsampling(
        frequency: CounterType[str],
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


def export_sorted_frequency_by_party(
        D_raw: CounterType[str],
        R_raw: CounterType[str],
        D_final: CounterType[str],
        R_final: CounterType[str],
        word_to_id: Dict[str, int],
        out_path: str
        ) -> None:

    def sort_creteria(tup: Tuple) -> Tuple[bool, float]:
        df, rf = tup[3], tup[4]  # final frequnecy
        if df != 0:
            ratio = rf / df
        else:
            ratio = rf / 1e-8
        nontrivial = df + rf > 100
        return (nontrivial, ratio)

    output = []
    for word in word_to_id:
        output.append(
            (word, D_raw[word], R_raw[word], D_final[word], R_final[word]))
    output.sort(
        key=sort_creteria,
        reverse=True)
    # output.sort(key=lambda tup: tup[4])
    with open(out_path, 'w') as out_file:
        out_file.write('Sorted by GOP/Dem Ratio    '
                       'Original (Dem, GOP)    '
                       'Sampled & Balanced [Dem, GOP]\n')
        for word, dr, rr, df, rf in output:
            raw_freq = f'({dr:,}, {rr:,})'
            final_freq = f'[{df:,}, {rf:,}]'
            out_file.write(f'{word:<30}{raw_freq:<20}{final_freq}\n')


def negative_sampling(
        subsampled_frequency: CounterType[str],
        id_to_word: Dict[int, str]
        ) -> List[float]:
    """
    A smoothed unigram distribution.
    A simplified case of Noise Contrasitive Estimate.
    where the seemingly aribitrary number of 0.75 is from the paper.
    """
    cumulative_freq = sum(freq ** 0.75 for freq in subsampled_frequency.values())
    sampling_dist: Dict[str, float] = {
        word: (freq ** 0.75) / cumulative_freq
        for word, freq in subsampled_frequency.items()
    }
    multinomial_dist_probs = [
        sampling_dist.get(id_to_word[word_id], 0)
        for word_id in range(len(id_to_word))
    ]
    return multinomial_dist_probs


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
        Dem_speeches: List[Document],
        GOP_speeches: List[Document]
        ) -> List[Document]:
    total = len(Dem_speeches) + len(GOP_speeches)
    print(f'Pre-balanced Dem speech ratio = {len(Dem_speeches) / total:.4%}')
    print(f'Pre-balanced GOP speech ratio = {len(GOP_speeches) / total:.4%}')
    minority = min(len(GOP_speeches), len(Dem_speeches))
    if len(GOP_speeches) > len(Dem_speeches):
        GOP_speeches = random.sample(GOP_speeches, k=minority)
        print(f'Balancing training data by sampling GOP to {minority:,} documents')
    else:
        Dem_speeches = random.sample(Dem_speeches, k=minority)
        print(f'Balancing training data by sampling Dem to {minority:,} documents')
    return Dem_speeches + GOP_speeches


def export_sentence_level_party_classifier(
        sessions: Iterable[int],
        output_dir: str,
        input_dir: str,
        subsampling_implementation: Optional[str],
        subsampling_threshold: float,
        min_word_freq: int,
        min_sentence_len: int
        ) -> None:
    """
    Document tokenized, with metadata per sentence.
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
    keep_prob = subsampling(
        raw_frequency, subsampling_implementation, subsampling_threshold)

    # num_words_exported = 0
    Dem_speeches: List[Document] = []
    GOP_speeches: List[Document] = []
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
                    #             Document(subsampled_word_ids, 0))
                    #     else:
                    #         num_GOP_speeches += 1
                    #         export_sentences.append(
                    #             Document(subsampled_word_ids, 1))

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
                                Document(subsampled_word_ids, 0))
                        else:
                            GOP_speeches.append(
                                Document(subsampled_word_ids, 1))

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

    export_sorted_frequency(
        raw_frequency, subsampled_frequency, min_word_freq,
        os.path.join(output_dir, 'vocabulary_frequency.txt'))

    train_data_path = os.path.join(output_dir, f'train_data.pickle')
    with open(train_data_path, 'wb') as export_file:
        pickle.dump(
            (word_to_id, id_to_word, subsampled_frequency, export_speeches),
            export_file, protocol=-1)
    print('All set.')


def export_word_level_party_classifier(
        sessions: Iterable[int],
        output_dir: str,
        input_dir: str,
        subsampling_implementation: Optional[str],
        subsampling_threshold: float,
        min_word_freq: int,
        min_sentence_len: int
        ) -> None:

    raw_frequency: CounterType[str] = Counter()
    for session in tqdm(sessions, desc='Building vocabulary'):
        for party in ('D', 'R'):
            underscored_path = os.path.join(
                input_dir, f'underscored_{party}{session}.txt')
            with open(underscored_path, 'r') as underscored_corpus:
                for speech in underscored_corpus:
                    speech = speech.translate(remove_punctutation)
                    words = speech.split()
                    raw_frequency.update(words)
    word_to_id, id_to_word = build_vocabulary(raw_frequency, min_word_freq)
    keep_prob = subsampling(
        raw_frequency, subsampling_implementation, subsampling_threshold)

    Dem_word_ids: List[Tuple[int, int]] = []
    GOP_word_ids: List[Tuple[int, int]] = []
    subsampled_frequency: CounterType[str] = Counter()
    for session in tqdm(sessions, desc=' Exporting Sessions'):
        for party in ('D', 'R'):
            underscored_path = os.path.join(
                input_dir, f'underscored_{party}{session}.txt')
            with open(underscored_path, 'r') as underscored_corpus:
                for speech in underscored_corpus:
                    speech = speech.translate(remove_punctutation)
                    subsampled_words = [
                        word for word in speech.split()
                        if (raw_frequency[word] > min_word_freq
                            and random.random() < keep_prob[word])
                    ]
                    subsampled_frequency.update(subsampled_words)
                    if party == 'D':
                        Dem_word_ids.extend(
                            [(0, word_to_id[word]) for word in subsampled_words])
                    else:
                        GOP_word_ids.extend(
                            [(1, word_to_id[word]) for word in subsampled_words])

    total = len(Dem_word_ids) + len(GOP_word_ids)
    print(f'Pre-balanced Dem word ratio = {len(Dem_word_ids) / total:.4%}')
    print(f'Pre-balanced GOP word ratio = {len(GOP_word_ids) / total:.4%}')
    minority = min(len(GOP_word_ids), len(Dem_word_ids))
    if len(GOP_word_ids) > len(Dem_word_ids):
        GOP_word_ids = random.sample(GOP_word_ids, k=minority)
        print(f'Balancing training data by sampling GOP to {minority:,}.')
    else:
        Dem_word_ids = random.sample(Dem_word_ids, k=minority)
        print(f'Balancing training data by sampling Dem to {minority:,}.')
    labels_and_word_ids = Dem_word_ids + GOP_word_ids

    validation_holdout = 50_000
    random.shuffle(labels_and_word_ids)
    train_data = labels_and_word_ids[validation_holdout:]
    valid_data = labels_and_word_ids[:validation_holdout]

    num_words = len(train_data)
    print(f'Number of training words = {len(train_data):,}')

    preview_path = os.path.join(output_dir, 'preview.txt')
    with open(preview_path, 'w') as preview:
        preview.write(f'vocabulary size = {len(word_to_id):,}\n')
        # preview.write(f'total number of words = {num_words_exported:,}\n')
        preview.write(f'number of words = {num_words:,}\n')
        preview.write(f'subsampling implementation = {subsampling_implementation}\n')
        preview.write(f'subsampling threshold = {subsampling_threshold}\n')
        preview.write(f'minimum word frequency = {min_word_freq}\n')
        # preview.write(f'\nPreview:\n')
        # for sentence in export_speeches:
        #     words = map(id_to_word.get, sentence.word_ids)  # type: ignore
        #     if sentence.party == 0:
        #         preview.write(f'D: {" ".join(words)}\n')
        #     else:
        #         preview.write(f'R: {" ".join(words)}\n')

    export_sorted_frequency(
        raw_frequency, subsampled_frequency, min_word_freq,
        os.path.join(output_dir, 'vocabulary_frequency.txt'))

    train_data_path = os.path.join(output_dir, f'train_data.pickle')
    with open(train_data_path, 'wb') as export_file:
        pickle.dump(
            (word_to_id, id_to_word, subsampled_frequency, train_data, valid_data),
            export_file, protocol=-1)
    print('All set.')


def export_skip_gram_corpus(
        sessions: Iterable[int],
        output_dir: str,
        input_dir: str,
        # output_num_chunks: int,
        subsampling_implementation: Optional[str],
        subsampling_threshold: float,
        min_word_freq: int,
        min_document_len: int
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
    keep_prob = subsampling(
        raw_frequency, subsampling_implementation, subsampling_threshold)

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
        if len(export_speech) < min_document_len:
            continue
        num_words_exported += len(export_speech)
        subsampled_frequency.update(export_speech)
        export_speech_in_word_ids = [word_to_id[word] for word in export_speech]
        export_speeches.append(export_speech_in_word_ids)

    export_sorted_frequency(
        raw_frequency, subsampled_frequency, min_word_freq,
        os.path.join(output_dir, 'vocabulary_frequency.txt'))

    vocab_path = os.path.join(output_dir, 'vocab.pickle')
    with open(vocab_path, 'wb') as vocab_file:
        pickle.dump((word_to_id, id_to_word, subsampled_frequency),
                    vocab_file, protocol=-1)

    train_data_path = os.path.join(output_dir, f'train_data.pickle')
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

    # sanity check negative sampling
    # negative_sampling_dist = negative_sampling(subsampled_frequency, id_to_word)
    # for word_id, word in id_to_word.items():
    #     print(f'{word}\t{subsampled_frequency[word]:,}'
    #           f'\t{negative_sampling_dist[word_id]:.2e}')
    print('All set.')


def export_adversarial(
        sessions: Iterable[int],
        output_dir: str,
        input_dir: str,
        subsampling_implementation: Optional[str],
        subsampling_threshold: float,
        min_word_freq: int,
        min_document_len: int
        ) -> None:
    raw_frequency: CounterType[str] = Counter()
    for session in tqdm(sessions, desc='Building vocabulary'):
        for party in ('D', 'R'):
            underscored_path = os.path.join(
                input_dir, f'underscored_{party}{session}.txt')
            with open(underscored_path, 'r') as underscored_corpus:
                for line in underscored_corpus:
                    line = line.translate(remove_punctutation)
                    words = line.split()
                    raw_frequency.update(words)
    word_to_id, id_to_word = build_vocabulary(raw_frequency, min_word_freq)
    keep_prob = subsampling(
        raw_frequency, subsampling_implementation, subsampling_threshold)

    Dem_speeches: List[Document] = []
    GOP_speeches: List[Document] = []
    Dem_raw_frequency: CounterType[str] = Counter()
    GOP_raw_frequency: CounterType[str] = Counter()

    for session in tqdm(sessions, desc=' Exporting Sessions'):
        for party in ('D', 'R'):
            underscored_path = os.path.join(
                input_dir, f'underscored_{party}{session}.txt')
            with open(underscored_path, 'r') as underscored_corpus:
                for line in underscored_corpus:
                    line = line.translate(remove_punctutation)
                    words = line.split()
                    subsampled_word_ids = [
                        word_to_id[word] for word in words
                        if (raw_frequency[word] > min_word_freq
                            and random.random() < keep_prob[word])
                    ]
                    if len(subsampled_word_ids) < min_document_len:
                        continue
                    if party == 'D':
                        Dem_raw_frequency.update(words)
                        Dem_speeches.append(Document(subsampled_word_ids, 0))
                    else:
                        GOP_raw_frequency.update(words)
                        GOP_speeches.append(Document(subsampled_word_ids, 1))
    export_speeches = balance_classes(Dem_speeches, GOP_speeches)

    Dem_word_count = 0
    GOP_word_count = 0
    Dem_frequency: CounterType[str] = Counter()
    GOP_frequency: CounterType[str] = Counter()
    combined_frequency: CounterType[str] = Counter()
    for speech in export_speeches:
        words = [id_to_word[word_id] for word_id in speech.word_ids]
        combined_frequency.update(words)
        if speech.party == 0:
            Dem_frequency.update(words)
            Dem_word_count += len(speech.word_ids)
        else:
            GOP_frequency.update(words)
            GOP_word_count += len(speech.word_ids)
    print(f'Pre-sampled vocab size = {len(word_to_id):,}')
    print(f'Post-sampled vocab size = {len(combined_frequency):,}')
    print(f'Post-balanced Dem word count = {Dem_word_count:,}')
    print(f'Post-balanced GOP word count = {GOP_word_count:,}')

    # Optionally rebuild vocabulary after sampling
    # word_to_id, id_to_word = build_vocabulary(
    #     combined_frequency, min_frequency=0)

    # validation_holdout = 50_000
    # train_data = export_speeches[validation_holdout:]
    # valid_data = export_speeches[:validation_holdout]

    preview_path = os.path.join(output_dir, 'preview.txt')
    with open(preview_path, 'w') as preview:
        preview.write(f'final vocabulary size = {len(word_to_id):,}\n')
        # preview.write(f'total number of words = {num_words_exported:,}\n')
        # preview.write(f'number of words = {num_words:,}\n')
        preview.write(f'subsampling implementation = {subsampling_implementation}\n')
        preview.write(f'subsampling threshold = {subsampling_threshold}\n')
        preview.write(f'minimum word frequency = {min_word_freq}\n')
        preview.write(f'\nPreview:\n')
        for speech in export_speeches[:100]:
            words = map(id_to_word.get, speech.word_ids)  # type: ignore
            preview.write(' '.join(words) + '\n')

    export_sorted_frequency(
        raw_frequency, combined_frequency, min_word_freq,
        os.path.join(output_dir, 'vocabulary_subsampled_frequency.txt'))
    export_sorted_frequency_by_party(
        Dem_raw_frequency, GOP_raw_frequency, Dem_frequency, GOP_frequency, word_to_id,
        os.path.join(output_dir, 'vocabulary_partisan_frequency.txt'))

    train_data_path = os.path.join(output_dir, f'train_data.pickle')
    with open(train_data_path, 'wb') as export_file:
        payload = (word_to_id, id_to_word, combined_frequency,
                   Dem_frequency, GOP_frequency, export_speeches)
        pickle.dump(payload, export_file, protocol=-1)
    print('All set.')


def skip_gram_from_plain_text(
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

    keep_prob = subsampling(
        raw_frequency, subsampling_implementation, subsampling_threshold)

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

    export_sorted_frequency(
        raw_frequency, subsampled_frequency, min_word_freq,
        os.path.join(output_dir, 'vocabulary_frequency.txt'))

    negative_sampling_dist = negative_sampling(subsampled_frequency, id_to_word)
    vocab_path = os.path.join(output_dir, 'vocab.pickle')
    with open(vocab_path, 'wb') as vocab_file:
        pickle.dump((word_to_id, id_to_word, negative_sampling_dist),
                    vocab_file, protocol=-1)
    print('All set.')


def main() -> None:
    during: Dict[str, Iterable[int]] = {
        # 'postwar': range(79, 112),
        'Truman': range(79, 83),
        'Eisenhower': range(83, 87),
        'Kennedy': range(87, 88),
        'Johnson': range(88, 91),
        'Nixon': range(91, 94),
        'Ford': range(94, 95),
        'Carter': range(95, 97),
        'Reagan': range(97, 101),
        'H.W.Bush': range(101, 103),
        'Clinton': range(103, 107),
        'W.Bush': range(107, 111),
        'Obama': range(111, 112)  # incomplete data in the Bound edition
    }

    # Just the corpus in plain text, no word_id. For external libraries
    # output_dir = '../../data/processed/plain_text/44_Obama_normalized'
    # underscored_dir = '../../data/interim/underscored_corpora_normalized'
    # os.makedirs(output_dir, exist_ok=True)
    # export_plain_text(sessions, output_dir, underscored_dir)

    # # tokenized corpus for skip-gram
    # output_dir = '../../data/processed/skip_gram/44_Obama_1e-5'
    # subsampling_implementation: Optional[str] = 'paper'
    # subsampling_threshold = 1e-5
    # underscored_dir = '../../data/interim/underscored_corpora'
    # min_word_freq = 15
    # min_speech_length = 5
    # os.makedirs(output_dir, exist_ok=True)
    # print(f'Reading {sessions}. Writing to {output_dir}')
    # export_skip_gram_corpus(
    #     sessions, output_dir, underscored_dir,  # output_num_chunks,
    #     subsampling_implementation, subsampling_threshold,
    #     min_word_freq, min_speech_length)

    # # Naive word-level party classifier
    # underscored_dir = '../../data/interim/underscored_corpora'
    # output_dir = '../../data/processed/word_classifier/2000s_1e-5'
    # subsampling_implementation: Optional[str] = 'paper'
    # subsampling_threshold = 1e-5
    # min_word_freq = 15
    # min_sentence_len = 5
    # os.makedirs(output_dir, exist_ok=True)
    # print(f'Writing to {output_dir}')
    # export_word_level_party_classifier(
    #     sessions, output_dir, underscored_dir,
    #     subsampling_implementation, subsampling_threshold,
    #     min_word_freq, min_sentence_len)

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

    # # Adversarial
    # sessions = during['Obama']
    # output_dir = '../../data/processed/adversarial/2000s'
    subsampling_implementation: Optional[str] = 'paper'
    subsampling_threshold = 1e-5
    underscored_dir = '../../data/interim/underscored_corpora'
    min_word_freq = 15
    min_speech_length = 20

    # os.makedirs(output_dir, exist_ok=True)
    # print(f'Reading {sessions}. Writing to {output_dir}')
    # export_adversarial(
    #     sessions, output_dir, underscored_dir,
    #     subsampling_implementation, subsampling_threshold,
    #     min_word_freq, min_speech_length)

    for presidency, session in during.items():
        output_dir = f'../../data/processed/adversarial/1e-5/{presidency}'
        os.makedirs(output_dir, exist_ok=True)
        tqdm.write(f'Reading {presidency}. Writing to {output_dir}')
        export_adversarial(
            session, output_dir, underscored_dir,
            subsampling_implementation, subsampling_threshold,
            min_word_freq, min_speech_length)



if __name__ == '__main__':
    main()
