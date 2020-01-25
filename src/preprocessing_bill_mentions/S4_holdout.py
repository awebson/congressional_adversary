import pickle
import random
import os
import multiprocessing as mp
from typing import Tuple, List, Iterable, Dict, Counter
from dataclasses import dataclass

from nltk.corpus import stopwords
from tqdm import tqdm

from search_bill_mentions import Bill, Speaker, Speech

random.seed(42)

procedural_words = {
    'yield', 'motion', 'order', 'ordered', 'quorum', 'roll', 'unanimous',
    'mr', 'madam', 'speaker', 'chairman', 'president', 'senator',
    'gentleman', 'colleague'}
stop_words = set(stopwords.words('english')).union(procedural_words)

sessions = range(97, 112)  # scraped up to 93
MIN_NUM_MENTIONS = 3
FIXED_SENT_LEN = 15
MIN_SENT_LEN = 5
MIN_WORD_FREQ = 15
DENO_LABEL = 'topic'
MAX_DEV_HOLDOUT = 100  # faux speeches per session
in_dir = '../../data/interim/bill_mentions/'
out_dir = '../../data/processed/bill_mentions/topic_deno/'
os.makedirs(out_dir, exist_ok=True)
print('Minimum number of mentions per bill =', MIN_NUM_MENTIONS)


@dataclass
class Sentence():
    words: List[str]
    deno: str
    cono: str
    # word_ids: List[int] = None
    # deno_id: int = None
    # cono_id: int = None


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


def process_sentences(session: int) -> Tuple[List[Sentence], List[Sentence]]:
    in_path = in_dir + f'underscored_{session}.pickle'
    with open(in_path, 'rb') as in_file:
        speeches, num_mentions = pickle.load(in_file)

    # Mark context speeches with bill denotation
    per_session_mention = 0
    for speech_index, speech in enumerate(speeches):
        if speech.mentions_bill is False:
            continue
        if num_mentions[speech.bill.title] < MIN_NUM_MENTIONS:
            continue

        per_session_mention += 1
        # NOTE hardcoded context_size
        # for i in range(speech_index - 2, speech_index + 8):
        for i in range(speech_index + 1, speech_index + 6):
            try:
                speeches[i].bill = speech.bill
            except IndexError:
                continue

    # Chop up sentences
    train_sent: List[Sentence] = []
    dev_sent: List[Sentence] = []
    random.shuffle(speeches)
    for speech in speeches:
        if speech.bill is None:
            continue
        if num_mentions[speech.bill.title] < MIN_NUM_MENTIONS:
            continue

        # deno = speech.bill.title
        deno = getattr(speech.bill, DENO_LABEL)  # either 'topic' or 'title'
        cono = speech.speaker.party
        if cono != 'D' and cono != 'R':  # skip independent members for now
            continue

        if len(dev_sent) < MAX_DEV_HOLDOUT:
            dev_sent += [
                Sentence(faux_sent, deno, cono)
                for faux_sent in faux_sent_tokenize(speech.text)]
        else:
            train_sent += [
                Sentence(faux_sent, deno, cono)
                for faux_sent in faux_sent_tokenize(speech.text)]

    check = sum([m for m in num_mentions.values() if m > 2])
    tqdm.write(
        f'Session {session}: '
        f'{per_session_mention} =?= {check} mentions above min, '
        f'{len(train_sent):,} faux sentences, {len(dev_sent)} dev holdout')
    return train_sent, dev_sent


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
        if word not in word_to_id and freq >= min_frequency:
            word_to_id[word] = next_vocab_id
            id_to_word[next_vocab_id] = word
            next_vocab_id += 1
    print(f'Vocabulary size = {len(word_to_id):,}')
    return word_to_id, id_to_word


train_sent = []
dev_sent = []
for train, dev in map(process_sentences, sessions):
    train_sent += [s for s in train]
    dev_sent += [s for s in dev]

print(f'len dev faux sent = {len(dev_sent)}')

# with mp.Pool(12) as team:
#     sentences = list(
#         sent  # flatten lists over multiple sessions
#         for session in team.imap_unordered(process_sentences, sessions)
#         for sent in session)

random.shuffle(train_sent)
word_freq = Counter((w for sent in train_sent for w in sent.words))
D_freq = Counter(
    (w for sent in train_sent if sent.cono == 'D'
     for w in sent.words))
R_freq = Counter(
    (w for sent in train_sent if sent.cono == 'R'
     for w in sent.words))
deno_freq = Counter((sent.deno for sent in train_sent))

# def partisan_freq(query):
#     d = D_freq[query]
#     r = R_freq[query]
#     return f'D: {d}, R: {r}, R-ratio: {r / (d + r):.2%}'

# import IPython
# IPython.embed()

word_to_id, id_to_word = build_vocabulary(word_freq, MIN_WORD_FREQ)
deno_to_id, id_to_deno = build_vocabulary(deno_freq, MIN_NUM_MENTIONS)
cono_to_id = {
    'D': 0,
    'R': 1}

def wrap(sentences):
    sent_word_ids: List[List[int]] = []
    deno_labels: List[int] = []
    cono_labels: List[int] = []
    for sent in sentences:
        sent_word_ids.append(
            [word_to_id[w] for w in sent.words if w in word_to_id])
        deno_labels.append(deno_to_id[sent.deno])
        cono_labels.append(cono_to_id[sent.cono])
    return sent_word_ids, deno_labels, cono_labels

t_swi, t_dl, t_cl = wrap(train_sent)
d_swi, d_dl, d_cl = wrap(dev_sent)

cucumbers = {
    'train_sent_word_ids': t_swi,
    'train_deno_labels': t_dl,
    'train_cono_labels': t_cl,
    'dev_sent_word_ids': d_swi,
    'dev_deno_labels': d_dl,
    'dev_cono_labels': d_cl,
    'word_to_id': word_to_id,
    'id_to_word': id_to_word,
    'deno_to_id': deno_to_id,
    'id_to_deno': id_to_deno,
    'Dem_freq': D_freq,
    'GOP_freq': R_freq,
}
out_path = out_dir + 'train_data.pickle'
with open(out_path, 'wb') as out_file:
    pickle.dump(cucumbers, out_file, protocol=-1)

inspect_label_path = out_dir + 'deno_labels.txt'
with open(inspect_label_path, 'w') as file:
    file.write('freq\tdeno_label\n')
    for d, f in sorted(deno_freq.items(), key=lambda t: t[1], reverse=True):
        file.write(f'{f}\t{d}\n')
