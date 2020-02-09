import pickle
import random
import os
from typing import Tuple, List, Iterable, Dict, DefaultDict, Counter
from dataclasses import dataclass

from nltk.corpus import stopwords
from tqdm import tqdm

random.seed(42)

procedural_words = {
    'yield', 'motion', 'order', 'ordered', 'quorum', 'roll', 'unanimous',
    'mr', 'madam', 'speaker', 'chairman', 'president', 'senator',
    'gentleman', 'colleague',
    'today', 'rise', 'rise today', 'pleased_to_introduce',
    'introducing_today', 'would_like'}

stop_words = set(stopwords.words('english')).union(procedural_words)

MIN_NUM_MENTIONS = 3
MIN_SENT_LEN = 5

sessions = range(97, 112)  # scraped up to 93; earlier ones unusally high bill mentions
in_dir = '../../data/interim/bill_mentions/'
out_path = '../../data/processed/bill_mentions/plain_corpus.txt'
print('Minimum number of mentions per bill =', MIN_NUM_MENTIONS)

total_num_words = 0
out_file = open(out_path, 'w')
for session in tqdm(sessions):
    in_path = os.path.join(in_dir, f'underscored_{session}.pickle')
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

    # random.shuffle(speeches)
    for speech in speeches:
        if speech.bill is None:
            continue
        if num_mentions[speech.bill.title] < MIN_NUM_MENTIONS:
            continue

        # deno = speech.bill.title
        # deno = getattr(speech.bill, DENO_LABEL)  # either 'topic' or 'title'
        cono = speech.speaker.party
        if cono != 'D' and cono != 'R':  # skip independent members for now
            continue

        words = [w for w in speech.text.split() if w not in stop_words]
        if len(words) < MIN_SENT_LEN:
            continue
        total_num_words += len(words)
        print(' '.join(words), file=out_file)

out_file.close()
print(f'Total number of words = {total_num_words:,}')
