import re
import os
import copy
import multiprocessing as mp
from typing import List, Iterable, Pattern

from tqdm import tqdm

punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'  # excluding underscore
remove_punctutation = str.maketrans('', '', punctuations)
contains_number = re.compile(r'\d')
REGEX_PATTERNS: List[Pattern]


def underscored_token(regex_match):
    return '_'.join(regex_match.group(0).split())


def process_speech(corpus: Iterable[str], patterns: List[Pattern]) -> List[str]:
    underscored: List[str] = []
    for line in corpus:
        line = line.lower()
        line = contains_number.sub('', line)
        for pattern in patterns:
            line = pattern.sub(underscored_token, line)
        line = line.translate(remove_punctutation)
        underscored.append(line)
    return underscored


def underscore_phrases(
        sessions: Iterable[int],
        output_dir: str,
        phrases_dir: str,
        partitioned_corpora_dir: str,
        manual_regex_patterns: List[Pattern],
        num_chunks: int,
        num_threads: int
        ) -> None:
    """
    Combine phrases from both parties to underscore.
    """
    for session in tqdm(sessions, desc='Sessions'):
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

        for party in ('D', 'R'):
            speeches: List[str] = []
            for chunk in tqdm(range(num_chunks), desc=f'{party} Chunks'):
                chunk_path = os.path.join(
                    partitioned_corpora_dir, f'{session}_{party}{chunk}.txt')
                with open(chunk_path) as corpus_chunk_file:
                    speeches += process_speech(corpus_chunk_file, patterns)

            output_path = os.path.join(
                output_dir, f'underscored_{party}{session}.txt')
            with open(output_path, 'w') as out_file:
                for speech in speeches:
                    out_file.write(speech)


def process_speech_mp(speech: str) -> str:
    speech = speech.lower()
    speech = contains_number.sub('', speech)
    for pattern in REGEX_PATTERNS:
        speech = pattern.sub(underscored_token, speech)
    speech = speech.translate(remove_punctutation)
    return speech


def underscore_phrases_mp(
        sessions: Iterable[int],
        output_dir: str,
        phrases_dir: str,
        partitioned_corpora_dir: str,
        manual_regex_patterns: List[Pattern],
        num_chunks: int,
        num_threads: int
        ) -> None:
    """
    Only underscore each party's own frequency phrases.

    Uses multiprocessing to speed up the regular expression substituion.
    """
    for session in tqdm(sessions, desc='Sessions'):
        speeches: List[str] = []
        for party in ('D', 'R'):
            phrases: List[str] = []
            phrases_path = os.path.join(phrases_dir, f'{session}_{party}.txt')
            with open(phrases_path) as in_file:
                for line in in_file:
                    _, phrase = line.split('\t')
                    phrases.append(phrase.strip())

            conglomerate_pattern = re.compile(
                f'({"|".join(map(re.escape, phrases))})')
            global REGEX_PATTERNS  # Sorry, but this simplifies multiprocessing
            REGEX_PATTERNS = copy.copy(manual_regex_patterns)  # resets the global patterns
            REGEX_PATTERNS.append(conglomerate_pattern)

            for chunk in tqdm(range(num_chunks), desc=f'{party} Chunks'):
                chunk_path = os.path.join(
                    partitioned_corpora_dir, f'{session}_{party}{chunk}.txt')
                with open(chunk_path) as corpus_chunk_file:
                    with mp.Pool(num_threads) as team:
                        speeches_per_chunk = team.imap_unordered(
                            process_speech_mp, corpus_chunk_file)
                        speeches += speeches_per_chunk

            output_path = os.path.join(
                output_dir, f'underscored_{party}{session}.txt')
            with open(output_path, 'w') as out_file:
                for speech in speeches:
                    out_file.write(speech)


def main() -> None:
    sessions = range(107, 111)
    num_chunks = 10
    num_threads = 40
    phrases_dir = '../../data/interim/aggregated_phrases'
    partitioned_corpora_dir = '../../data/interim/partitioned_corpora'
    output_dir = '../../data/interim/underscored_corpora'
    manual_regex_patterns = [
        re.compile(r'the (\w+ \b){1,2}(bill|act|amendment|reform)')
    ]
    print(f'Underscoring {sessions}. Writing to {output_dir}')
    underscore_phrases(
        sessions, output_dir, phrases_dir, partitioned_corpora_dir,
        manual_regex_patterns, num_chunks, num_threads)

if __name__ == '__main__':
    main()
