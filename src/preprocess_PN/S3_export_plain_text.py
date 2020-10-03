import pickle
import random
from pathlib import Path
from typing import Tuple, List, Dict, Counter, Optional

from tqdm import tqdm

from data import Sentence, LabeledDoc

random.seed(42)

def main(
        in_dir: Path,
        out_dir: Path,
        min_frequency: int,
        min_sent_len: int,
        max_sent_len: int,
        conserve_RAM: bool = False
        ) -> None:
    Path.mkdir(out_dir, parents=True, exist_ok=True)
    preview = open(out_dir / f'preview.txt', 'w')
    print(f'Min word frequency = {min_frequency}', file=preview)
    print(f'Min sentence length = {min_sent_len}', file=preview)
    print(f'Max sentence length = {max_sent_len}', file=preview)

    corpus: List[LabeledDoc] = []
    print('Loading multi-word expression underscored pickle...')
    with open(in_dir / f'MWE_underscored.pickle', 'rb') as in_file:
        corpus += pickle.load(in_file)

    norm_freq: Counter[str] = Counter()
    for doc in tqdm(corpus, desc='Counting UNKs'):
        for sent in doc.sentences:
            norm_freq.update(sent.underscored_tokens)
    print(f'Noramlized vocabulary size = {len(norm_freq):,}', file=preview)
    print(f'Number of words = {sum(freq for freq in norm_freq.values()):,}',
          file=preview)

    # Filter counter with MIN_FREQ and count UNK
    UNK_filtered_freq: Counter[str] = Counter()
    for key, val in norm_freq.items():
        if val >= min_frequency:
            UNK_filtered_freq[key] = val
        else:
            UNK_filtered_freq['<UNK>'] += val
    print(f'UNK-filtered vocabulary size = {len(UNK_filtered_freq):,}', file=preview)

    final_freq: Counter[str] = Counter()
    for doc in tqdm(corpus, desc='Filtering by sentence length'):
        for sent in doc.sentences:
            sent.subsampled_tokens = [
                token if token in UNK_filtered_freq else '<UNK>'
                for token in sent.underscored_tokens]
            if len(sent.subsampled_tokens) >= min_sent_len:
                if len(sent.subsampled_tokens) <= max_sent_len:
                    final_freq.update(sent.subsampled_tokens)
                else:  # NOTE truncate long sentences
                    sent.subsampled_tokens = sent.subsampled_tokens[:max_sent_len]
                    final_freq.update(sent.subsampled_tokens)
            else:  # discard short sentences
                sent.subsampled_tokens = None
            if conserve_RAM:
                del sent.normalized_tokens
                del sent.underscored_tokens
        # End looping sentences
        doc.sentences = [
            sent for sent in doc.sentences
            if sent.subsampled_tokens is not None]
    # End looping documents
    corpus = [doc for doc in corpus if len(doc.sentences) > 0]

    print(f'Final vocabulary size = {len(final_freq):,}', file=preview)
    print(f'Sentence-length filtered number of words = '
          f'{sum(freq for freq in final_freq.values()):,}',
          file=preview)

    # random.shuffle(corpus)
    # # Print one sentence per line
    # with open(out_dir / 'train.txt', 'w') as train_file:
    #     for doc in tqdm(corpus, desc=f'Writing to {out_dir}'):
    #         for sent in doc.sentences:
    #             print(' '.join(sent.subsampled_tokens), file=train_file)
    # Print one document per line
    with open(out_dir / 'train.txt', 'w') as train_file:
        for doc in tqdm(corpus, desc=f'Writing to {out_dir}'):
            for sent in doc.sentences:
                print(' '.join(sent.subsampled_tokens), end=' ', file=train_file)
            train_file.write('\n')


    # Print out vocabulary & some random sentences for sanity check
    docs = random.sample(corpus, 100)
    preview.write('\n')
    for doc in docs:
        sent = doc.sentences[0]
        if not conserve_RAM:
            print(sent.tokens, file=preview)
            print(sent.normalized_tokens, file=preview)
            print(sent.subsampled_tokens, file=preview, end='\n\n')
        else:
            print(sent.numerical_tokens, file=preview)
            # print(vars(doc), end='\n\n', file=preview)
    preview.write('\n\nword\tsentence_length_filtered_freq\n')
    for key, val in final_freq.most_common():
        print(f'{val:,}:\t{key}', file=preview)
    preview.close()
    print('All set!')


if __name__ == '__main__':
    main(
        in_dir=Path('../../data/interim/news'),
        out_dir=Path('../../data/ready/PN_plain'),
        min_frequency=30,
        min_sent_len=5,
        max_sent_len=20)
