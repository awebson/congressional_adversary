import re
import os
import copy
import pickle
import multiprocessing as mp
from typing import List, Iterable, Pattern

from tqdm import tqdm
# from nltk.corpus import stopwords

from search_bill_mentions import Bill, Speaker, Speech

# procedural_words = {
#     'yield', 'motion', 'order', 'ordered', 'quorum', 'roll', 'unanimous',
#     'mr', 'madam', 'speaker', 'chairman', 'president', 'senator',
#     'gentleman', 'colleague'}
# stopwords = set(stopwords.words('english')).union(procedural_words)
# stopwords_pattern = re.compile(f'({"|".join(stopwords)})')

remove_punctutations = str.maketrans(
    '', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~')  # excluding underscore
remove_numbers = str.maketrans('', '', '0123456789')

phrases_dir = '../../data/interim/aggregated_phrases'
bill_mentions_dir = '../../data/interim/bill_mentions/daily'
output_dir = '../../data/interim/bill_mentions/daily'
manual_regex_patterns = [
    re.compile(r'the (\w+ \b){1,2}(bill|act|amendment|reform)')
]


def underscored_token(regex_match):
    return '_'.join(regex_match.group(0).split())


def underscore_phrases(session: int) -> None:
    """
    Combine phrases from both parties to underscore.
    """
    phrases: List[str] = []
    phrases_path = os.path.join(phrases_dir, f'{session}.txt')
    with open(phrases_path) as in_file:
        for line in in_file:
            _, _, phrase = line.split('\t')
            phrases.append(phrase.strip())
    phrases = map(re.escape, phrases)  # type: ignore
    conglomerate_pattern = re.compile(f'({"|".join(phrases)})')  # type: ignore
    patterns = copy.copy(manual_regex_patterns)
    patterns.append(conglomerate_pattern)

    # bill mentions RegEx results
    cache_path = os.path.join(bill_mentions_dir, f'cache_{session}.pickle')
    with open(cache_path, 'rb') as cache_file:
        speeches, num_mentions = pickle.load(cache_file)

    for speech in speeches:
        text = speech.text.lower().translate(remove_numbers)
        # text = stopwords_pattern.sub('', text)
        for pattern in patterns:
            text = pattern.sub(underscored_token, text)
        text = text.translate(remove_punctutations)
        speech.text = text

    output_path = os.path.join(output_dir, f'underscored_{session}.pickle')
    with open(output_path, 'wb') as pickle_file:
        pickle.dump((speeches, num_mentions), pickle_file)


def main() -> None:
    sessions = range(97, 115)  # 93
    # num_threads = 12
    # phrases_dir = '../../data/interim/aggregated_phrases'
    # bill_mentions_dir = '../../data/interim/bill_mentions'
    # output_dir = '../../data/interim/bill_mentions_underscored'
    # manual_regex_patterns = [
    #     re.compile(r'the (\w+ \b){1,2}(bill|act|amendment|reform)')
    # ]
    # print(f'Underscoring {sessions}. Writing to {output_dir}')

    with mp.Pool(12) as team:
        _ = tuple(tqdm(team.imap_unordered(underscore_phrases, sessions),
                       total=len(sessions)))

if __name__ == '__main__':
    main()
