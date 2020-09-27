import pickle
import random
from pathlib import Path
from typing import Tuple, List, Iterable, Dict, Counter
from dataclasses import dataclass

from nltk.corpus import stopwords
from tqdm import tqdm

from data import GroundedWord

random.seed(42)

procedural_words = {
    'yield', 'motion', 'order', 'ordered', 'quorum', 'roll', 'unanimous',
    'mr', 'madam', 'speaker', 'chairman', 'president', 'senator',
    'gentleman', 'colleague',
    'today', 'rise', 'rise today', 'pleased_to_introduce',
    'introducing_today', 'would_like'
}
discard = set(stopwords.words('english')).union(procedural_words)

sessions = range(97, 112)  # scraped up to 93
MIN_NUM_MENTIONS = 3
FIXED_SENT_LEN = 15
MIN_SENT_LEN = 5
MIN_WORD_FREQ = 15
NUM_CONTEXT_SPEECHES = 3
NUM_DEV_HOLDOUT = 100  # faux sentences per session

EVAL_MIN_FREQ = 100
EVAL_R_THRESHOLDS = (0.6, 0.7, 0.75, 0.8, 0.9)
EVAL_NUM_RANDOM_SAMPLE = 500

# DENO_LABEL = 'topic'
DENO_LABEL = 'title'

in_dir = Path('../../data/interim/bill_mentions/')
# out_dir = Path(f'../../data/ready/CR_topic_context{NUM_CONTEXT_SPEECHES}')
out_dir = Path(f'../../data/ready/CR_bill_context{NUM_CONTEXT_SPEECHES}')
Path.mkdir(out_dir, parents=True, exist_ok=True)

print('Writing to to ', out_dir)
log = open(out_dir / 'log.txt', 'w')
log.write(f'Num of context speeches per bill mention = {NUM_CONTEXT_SPEECHES}\n')
log.write(f'Minimum number of mentions per bill = {MIN_NUM_MENTIONS}\n')


@dataclass
class Sentence():
    words: List[str]
    deno: str
    cono: str


def faux_sent_tokenize(line: str) -> Iterable[List[str]]:
    """discard procedural words and punctuations"""
    words = [w for w in line.split() if w not in discard]
    start_index = 0
    while (start_index + FIXED_SENT_LEN) < (len(words) - 1):
        yield words[start_index:start_index + FIXED_SENT_LEN]
        start_index += FIXED_SENT_LEN

    trailing_words = words[start_index:-1]
    if len(trailing_words) >= MIN_SENT_LEN:
        yield trailing_words


def process_sentences(session: int) -> Tuple[List[Sentence], List[Sentence]]:
    in_path = in_dir / f'underscored_{session}.pickle'
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
        for i in range(speech_index + 1,
                       speech_index + 1 + NUM_CONTEXT_SPEECHES):
            try:
                speeches[i].bill = speech.bill
            except IndexError:
                continue

    # Chop up sentences
    sentences: List[Sentence] = []
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

        sentences += [
            Sentence(faux_sent, deno, cono)
            for faux_sent in faux_sent_tokenize(speech.text)]

    random.shuffle(sentences)
    dev = sentences[:NUM_DEV_HOLDOUT]
    train = sentences[NUM_DEV_HOLDOUT:]

    check = sum([m for m in num_mentions.values() if m > 2])
    progress = (
        f'Session {session}: '
        f'{per_session_mention} bill mentions above min, '
        f'{len(train):,} sentences, {len(dev)} dev holdout')
    print(progress)
    print(progress, file=log)
    return train, dev


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
    print(f'Vocabulary size = {len(word_to_id):,}', file=log)
    return word_to_id, id_to_word


# Sorry for the lack of main()
train_sent: List[Sentence] = []
dev_sent: List[Sentence] = []
for train, dev in map(process_sentences, sessions):
    train_sent += [s for s in train]
    dev_sent += [s for s in dev]
print(f'Number of train sentences = {len(train_sent):,}', file=log)
print(f'Number of dev sentences = {len(dev_sent):,}', file=log)

word_freq = Counter((w for sent in train_sent for w in sent.words))
sent_deno_freq = Counter((sent.deno for sent in train_sent))

ground: Dict[str, GroundedWord] = {}
for sent in train_sent:
    for word in sent.words:
        if word not in ground:
            ground[word] = GroundedWord(
                text=word,
                deno=Counter({sent.deno: 1}),
                cono=Counter({sent.cono: 1}))
        else:
            ground[word].deno[sent.deno] += 1
            ground[word].cono[sent.cono] += 1


# EVAL_R_THRESHOLD = 0.55
# EVAL_D_THRESHOLD = 1 - EVAL_R_THRESHOLD
random_eval_words = set()
for gw in ground.values():
    gw.majority_deno = gw.deno.most_common(1)[0][0]
    gw.majority_cono = gw.cono.most_common(1)[0][0]
    gw.freq = sum(gw.cono.values())
    gw.R_ratio = gw.cono['R'] / gw.freq
    if gw.freq >= EVAL_MIN_FREQ:
        random_eval_words.add(gw.text)
random_eval_words = random.sample(random_eval_words, EVAL_NUM_RANDOM_SAMPLE)
with open(out_dir / f'eval_words_random.txt', 'w') as file:
    file.write('\n'.join(random_eval_words))


for R_threshold in EVAL_R_THRESHOLDS:
    D_threshold = 1 - R_threshold
    partisan_eval_words = []
    for gw in ground.values():
        if gw.freq >= EVAL_MIN_FREQ:
            if gw.R_ratio >= R_threshold or gw.R_ratio <= D_threshold:
                partisan_eval_words.append(gw)
    print(f'{len(partisan_eval_words)} partisan eval words '
          f'with R_threshold = {R_threshold}', file=log)

    out_path = out_dir / f'inspect_{R_threshold}_partisan.tsv'
    with open(out_path, 'w') as file:
        print('word\tfreq\tR_ratio', file=file)
        for gw in partisan_eval_words:
            print(gw.text, gw.freq, gw.R_ratio, sep='\t', file=file)

    random.shuffle(partisan_eval_words)
    mid = len(partisan_eval_words) // 2
    with open(out_dir / f'{R_threshold}partisan_dev_words.txt', 'w') as file:
        for gw in partisan_eval_words[:mid]:
            print(gw.text, file=file)
    with open(out_dir / f'{R_threshold}partisan_test_words.txt', 'w') as file:
        for gw in partisan_eval_words[mid:]:
            print(gw.text, file=file)

word_to_id, id_to_word = build_vocabulary(
    word_freq, MIN_WORD_FREQ, add_special_tokens=True)
deno_to_id, id_to_deno = build_vocabulary(
    sent_deno_freq, MIN_NUM_MENTIONS, add_special_tokens=False)
cono_to_id = {
    'D': 0,
    'R': 1}

# NOTE filter UNK here?

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
    'ground': ground
}
out_path = out_dir / 'train_data.pickle'
with open(out_path, 'wb') as out_file:
    pickle.dump(cucumbers, out_file, protocol=-1)

inspect_label_path = out_dir / 'sentence_deno_freq.txt'
with open(inspect_label_path, 'w') as file:
    file.write('freq\tdeno_label\n')
    for d, f in sorted(sent_deno_freq.items(), key=lambda t: t[1], reverse=True):
        file.write(f'{f}\t{d}\n')

log.close()
