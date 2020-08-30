import pickle
import random
import os
from typing import Tuple, List, Iterable, Dict, DefaultDict, Counter
from dataclasses import dataclass

from nltk.corpus import stopwords
from tqdm import tqdm

from data import GroundedWord, Sentence, LabeledDoc

random.seed(42)

procedural_words = {
    'yield', 'motion', 'order', 'ordered', 'quorum', 'roll', 'unanimous',
    'mr', 'madam', 'speaker', 'chairman', 'president', 'senator',
    'gentleman', 'colleague',
    'today', 'rise', 'rise today', 'pleased_to_introduce',
    'introducing_today', 'would_like'
}
stop_words = set(stopwords.words('english')).union(procedural_words)

sessions = range(97, 112)  # scraped up to 93
MIN_NUM_MENTIONS = 3
FIXED_SENT_LEN = 15
MIN_SENT_LEN = 5
MIN_WORD_FREQ = 15
DENO_LABEL = 'topic'
NUM_CONTEXT_SPEECHES = 3
MAX_DEV_HOLDOUT = 100  # faux speeches per session
in_dir = '../../data/interim/bill_mentions/'
out_dir = '../../data/processed/bill_mentions/debug'
os.makedirs(out_dir, exist_ok=True)
print('Minimum number of mentions per bill =', MIN_NUM_MENTIONS)


# @dataclass
# class Sentence():
#     words: List[str]
#     deno: str
#     cono: str
#     # word_ids: List[int] = None
#     # deno_id: int = None
#     # cono_id: int = None


def faux_sent_tokenize(line: str) -> Iterable[List[str]]:
    """discard procedural words and punctuations"""
    words = [w for w in line.split() if w not in stop_words]
    start_index = 0
    while (start_index + FIXED_SENT_LEN) < (len(words) - 1):
        yield words[start_index:start_index + FIXED_SENT_LEN]
        start_index += FIXED_SENT_LEN

    trailing_words = words[start_index:-1]
    if len(trailing_words) >= MIN_SENT_LEN:
        yield trailing_words


def process_sentences(session: int) -> List[LabeledDoc]:
    in_path = os.path.join(in_dir, f'underscored_{session}.pickle')
    with open(in_path, 'rb') as in_file:
        speeches, num_mentions = pickle.load(in_file)

    # Mark context speeches with bill denotation
    mentions_per_session = 0
    for speech_index, speech in enumerate(speeches):
        if speech.mentions_bill is False:
            continue
        if num_mentions[speech.bill.title] < MIN_NUM_MENTIONS:
            continue

        mentions_per_session += 1
        for i in range(speech_index + 1,
                       speech_index + 1 + NUM_CONTEXT_SPEECHES):
            try:
                speeches[i].bill = speech.bill
            except IndexError:
                continue

    # Chop up sentences
    random.shuffle(speeches)
    docs: List[LabeledDoc] = []
    uid = 0
    for speech in speeches:
        if speech.bill is None:
            continue
        if num_mentions[speech.bill.title] < MIN_NUM_MENTIONS:
            continue

        deno = getattr(speech.bill, DENO_LABEL)  # either 'topic' or 'title'
        cono = speech.speaker.party
        if cono != 'D' and cono != 'R':  # skip independent members for now
            continue

        # if len(dev_sent) < MAX_DEV_HOLDOUT:
        #     dev_sent += [
        #         Sentence(faux_sent, deno, cono)
        #         for faux_sent in faux_sent_tokenize(speech.text)]

        docs.append(
            LabeledDoc(
                uid=f'{session}_{uid}',
                title=None,
                url=None,
                party=cono,
                referent=deno,
                text=None,
                date=None,
                sentences=[Sentence(tokens=faux_sent)
                           for faux_sent in faux_sent_tokenize(speech.text)]
            ))
        uid += 1

    check = sum([m for m in num_mentions.values() if m > 2])
    tqdm.write(
        f'Session {session}: '
        # f'{mentions_per_session} =?= {check} mentions above min, '
        # f'{len(docs):,} faux sentences')  {len(dev_sent)} dev holdout')
    )
    return docs


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


def main(
        conserve_RAM: bool
        ) -> None:
    docs: List[LabeledDoc] = []
    # dev_sent: List[Sentence] = []
    for train in map(process_sentences, sessions):
        docs += [s for s in train]
        # dev_sent += [s for s in dev]
    # print(f'len dev faux sent = {len(dev_sent)}')


    random.shuffle(docs)
    word_freq = Counter(
        (w for doc in docs for sent in doc.sentences for w in sent.tokens))
    sent_deno_freq = Counter((doc.title for doc in docs))

    grounding: Dict[str, Counter[str]] = DefaultDict(Counter)  # word -> Counter[deno/cono]
    for sent in docs:
        for word in sent.tokens:
            grounding[word][sent.deno] += 1
            grounding[word][sent.cono] += 1
    # # counts['freq'] = word_freq

    ground: Dict[str, GroundedWord] = {}

    # for word, ground in grounding.items():
    #     majority_deno = ground.most_common(3)  # HACK
    #     for guess, _ in majority_deno:
    #         if guess not in ('D', 'R'):
    #             ground['majority_deno'] = guess  # type: ignore
    #             break
    #     assert ground['D'] + ground['R'] == word_freq[word]
    #     ground['freq'] = word_freq[word]
    #     ground['R_ratio'] = ground['R'] / word_freq[word]  # type: ignore

    word_to_id, id_to_word = build_vocabulary(
        word_freq, MIN_WORD_FREQ, add_special_tokens=True)
    deno_to_id, id_to_deno = build_vocabulary(
        sent_deno_freq, MIN_NUM_MENTIONS, add_special_tokens=False)
    cono_to_id = {
        'D': 0,
        'R': 1}

    for doc in docs:
        doc.party = cono_to_id[doc.party]
        doc.referent = deno_to_id[doc.referent]
        for sent in doc.sentences:
            sent.numerical_tokens = [word_to_id[w]
                                     for w in sent.tokens
                                     if w in word_to_id]
            if conserve_RAM is True:
                sent.tokens = None

    # # Helper for negative sampling
    # cumulative_freq = sum(freq ** 0.75 for freq in word_freq.values())
    # negative_sampling_probs: Dict[int, float] = {
    #     word_to_id[word]: (freq ** 0.75) / cumulative_freq
    #     for word, freq in word_freq.items()}
    # negative_sampling_probs: List[float] = [
    #     # negative_sampling_probs[word_id]  # strict
    #     negative_sampling_probs.get(word_id, 0)  # prob = 0 if missing vocab
    #     for word_id in range(len(word_to_id))]

    cucumbers = {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'numericalize_cono': cono_to_id,
        'numericalize_deno': deno_to_id,
        'ground': ground,
        # 'negative_sampling_probs': negative_sampling_probs,
        'documents': docs,
    }
    out_path = os.path.join(out_dir, 'train_data.pickle')
    with open(out_path, 'wb') as out_file:
        pickle.dump(cucumbers, out_file, protocol=-1)

    inspect_label_path = os.path.join(out_dir, 'doc_deno_freq.txt')
    with open(inspect_label_path, 'w') as file:
        file.write('freq\tdeno_label\n')
        for d, f in sorted(sent_deno_freq.items(), key=lambda t: t[1], reverse=True):
            file.write(f'{f}\t{d}\n')


if __name__ == "__main__":
    main(conserve_RAM=True)
