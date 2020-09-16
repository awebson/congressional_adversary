import pickle
import os
from typing import Tuple, List, Iterable, Dict, DefaultDict, Counter

from nltk.corpus import stopwords
from tqdm import tqdm


procedural_words = {
    'yield', 'motion', 'order', 'ordered', 'quorum', 'roll', 'unanimous',
    'mr', 'madam', 'speaker', 'chairman', 'president', 'senator',
    'gentleman', 'colleague',
    'today', 'rise', 'rise today', 'pleased_to_introduce',
    'introducing_today', 'would_like'}

discard = set(stopwords.words('english')).union(procedural_words)

MIN_SENT_LEN = 5
NUM_CONTEXT_SPEECHES = 3

sessions = range(97, 112)  # scraped up to 93; earlier ones unusally high bill mentions
in_dir = '../../data/interim/bill_mentions/'
out_path = '../../data/processed/plain_97.txt'

total_num_words = 0
out_file = open(out_path, 'w')
vocab = set()
for session in tqdm(sessions):
    in_path = os.path.join(in_dir, f'underscored_{session}.pickle')
    with open(in_path, 'rb') as in_file:
        speeches, num_mentions = pickle.load(in_file)

    for speech in speeches:
        cono = speech.speaker.party
        if cono != 'D' and cono != 'R':  # skip independent members for now
            continue

        words = [w for w in speech.text.split() if w not in discard]
        if len(words) < MIN_SENT_LEN:
            continue
        vocab.update(words)
        total_num_words += len(words)
        print(' '.join(words), file=out_file)

out_file.close()
print(f'Unfiltered vocabulary size = {len(vocab):,}')
print(f'Total number of words = {total_num_words:,}')
