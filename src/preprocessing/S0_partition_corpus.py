import json
import csv
import os
from collections import Counter
from typing import Set, Tuple, List, Dict, Iterable

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def partition_corpus(
        session: int,
        corpus_path: str,
        metadata_path: str,
        output_dir: str,
        min_speech_length: int,
        num_chunks: int,
        input_encoding: str
        ) -> Tuple[int, int, int, int]:
    """
    Partition a large corpus into multiple smaller chunks.

    Albert Webson [21:27]
    This [grid] job since noon is the full 200 MB. I just submitted a quartered
    50 MB job a couple hours ago. If that still doesn’t work, I will octo-...
    How is there no verb for “divide into 8 parts”? There is “decimate”,
    although it doesn’t mean “divide into 10 parts” because its etymology is
    the Roman punishment for mutiny by executing every one in ten soldiers.

    Ellie Pavlick [21:30]
    i officially declare “decimate” means “split the job into 10 parallel jobs.”
    “your parsing job is slow? did you try decimating it?”
    """

    def decimate(speeches: List[str]) -> Iterable[List[str]]:
        chunk_size = len(speeches) // num_chunks
        speech_index = 0
        chunk_index = 0
        while chunk_index <= num_chunks - 2:
            yield speeches[speech_index:speech_index + chunk_size]
            speech_index += chunk_size
            chunk_index += 1
        yield speeches[speech_index:-1]

    speech_id_to_party: Dict[str, str] = dict()
    with open(metadata_path) as metadata_file:
        reader = csv.DictReader(metadata_file, delimiter='|')
        for speaker_data in reader:
            if speaker_data['nonvoting'] == 'nonvoting':
                continue
            speech_id_to_party[speaker_data['speech_id']] = speaker_data['party']

    dem_corpus: List[str] = []
    gop_corpus: List[str] = []
    speech_count = 0
    word_count = 0
    short_speech_count = 0
    missing_metadata_count = 0

    with open(corpus_path, encoding=input_encoding) as corpus_file:
        corpus_file.readline()  # discard header line
        for line in corpus_file:
            try:
                speech_id, speech = line.split('|')
            except ValueError:  # from spliting line with '|'
                continue
            speech_count += 1
            if speech_id not in speech_id_to_party:
                missing_metadata_count += 1
                continue
            num_words = len(speech.split())
            if num_words < min_speech_length:
                short_speech_count += 1
                continue
            word_count += num_words
            party = speech_id_to_party[speech_id]
            if party == 'D':
                dem_corpus.append(speech)
            elif party == 'R':
                gop_corpus.append(speech)

    for chunk_index, corpus_chunk in enumerate(decimate(dem_corpus)):
        out_path = os.path.join(output_dir, f'{session}_D{chunk_index}.txt')
        with open(out_path, 'w') as out_file:
            for speech in corpus_chunk:
                out_file.write(speech)

    for chunk_index, corpus_chunk in enumerate(decimate(gop_corpus)):
        out_path = os.path.join(output_dir, f'{session}_R{chunk_index}.txt')
        with open(out_path, 'w') as out_file:
            for speech in corpus_chunk:
                out_file.write(speech)

    return speech_count, word_count, short_speech_count, missing_metadata_count


def partition_jsonl_corpus(
        corpus_path: str,
        output_dir: str,
        num_chunks: int,
        input_encoding: str
        ) -> None:
    """
    For the corpus scraped from the Presidency Project at UC Santa Barbara.
    Partition a single JSONL corpus into multiple smaller chunks.

    No checking minimum speech length, since they're all reasonably long.
    """

    def decimate(speeches: List[str]) -> Iterable[List[str]]:
        chunk_size = len(speeches) // num_chunks
        speech_index = 0
        chunk_index = 0
        while chunk_index <= num_chunks - 2:
            yield speeches[speech_index:speech_index + chunk_size]
            speech_index += chunk_size
            chunk_index += 1
        yield speeches[speech_index:-1]

    polisci101 = {
        'Harry S. Truman': 0,
        'Dwight D. Eisenhower': 1,
        'John F. Kennedy': 0,
        'Lyndon B. Johnson': 0,
        'Richard Nixon': 1,
        'Gerald R. Ford': 1,
        'Jimmy Carter': 0,
        'Ronald Reagan': 1,
        'George Bush': 1,
        'William J. Clinton': 0,
        'George W. Bush': 1,
        'Barack Obama': 0,
        'Donald J. Trump': 1
    }

    dem_corpus: List[str] = []
    gop_corpus: List[str] = []
    speech_count = 0
    word_count = 0

    skipped: Counter[str] = Counter()
    with open(corpus_path, 'r') as jsonl_file:
        for line in jsonl_file:
            json_obj = json.loads(line)
            name = json_obj['person']
            if name not in polisci101:
                skipped[name] += 1
                continue
            try:
                speech = '\n'.join(json_obj['speech'])
            except KeyError:
                continue
            party = polisci101[name]
            if party == 0:
                dem_corpus.append(speech)
            else:
                gop_corpus.append(speech)
    print('Skipped the following undefined people: ', skipped)

    for chunk_index, corpus_chunk in enumerate(decimate(dem_corpus)):
        out_path = os.path.join(output_dir, f'D{chunk_index}.txt')
        with open(out_path, 'w') as out_file:
            for speech in corpus_chunk:
                out_file.write(speech)

    for chunk_index, corpus_chunk in enumerate(decimate(gop_corpus)):
        out_path = os.path.join(output_dir, f'R{chunk_index}.txt')
        with open(out_path, 'w') as out_file:
            for speech in corpus_chunk:
                out_file.write(speech)


def speech_length_histogram(
        sessions: Iterable[int],
        histogram_upper_bound: int = 50,
        metadata_of_interest: Set[str] = {'party', 'chamber', 'gender', 'state'},
        identities: Set[str] = {'Dem', 'GOP', 'Senate', 'House', 'Male', 'Female'}
        ) -> None:

    speeches_length: defaultdict[str, List[int]] = defaultdict(list)
    for session_index in tqdm(sessions):
        metadata: Dict[str, Dict[str, str]] = dict()
        metadata_path = f'corpora/bound/{session_index:0>3d}_SpeakerMap.txt'
        with open(metadata_path) as metadata_file:
            reader = csv.DictReader(metadata_file, delimiter='|')
            for speaker_data in reader:
                if speaker_data['nonvoting'] == 'nonvoting':
                    continue
                speaker: Dict[str, str] = {
                    attribute: speaker_data[attribute]
                    for attribute in metadata_of_interest}
                metadata[speaker_data['speech_id']] = speaker

        speech_count = 0
        missing_metadata_count = 0
        corpus_path = f'corpora/bound/speeches_{session_index:0>3d}.txt'
        with open(corpus_path, encoding=input_encoding) as corpus_file:
            corpus_file.readline()  # discard header line
            for line in corpus_file:
                try:
                    speech_id, speech = line.split('|')
                    speech_count += 1
                    if speech_id not in metadata:
                        missing_metadata_count += 1
                        continue

                    speaker = metadata[speech_id]
                    party = speaker['party']
                    chamber = speaker['chamber']
                    gender = speaker['gender']
                    state = speaker['state']

                    speech_length = len(speech.split())

                    speeches_length[state].append(speech_length)
                    if party == 'D':
                        speeches_length['Dem'].append(speech_length)
                    elif party == 'R':
                        speeches_length['GOP'].append(speech_length)
                    # else:
                    #     print('Spoiler effect:', party)

                    if chamber == 'S':
                        speeches_length['Senate'].append(speech_length)
                    elif chamber == 'H':
                        speeches_length['House'].append(speech_length)
                    else:
                        print('Bicameralism is bad enough:', chamber)

                    if gender == 'M':
                        speeches_length['Male'].append(speech_length)
                    elif gender == 'F':
                        speeches_length['Female'].append(speech_length)
                    else:
                        print('Nonbinary:')
                except ValueError:  # from spliting line with '|'
                    continue
        missing_metadata_ratio = missing_metadata_count / speech_count
        tqdm.write(f'{missing_metadata_ratio:.2%} speeches in {corpus_path} '
                   'are missing metadata and excluded from the output corpus.')

    for metadata_name in identities:
        bounded_lengths = [length for length in speeches_length[metadata_name]
                           if length < histogram_upper_bound]

        if len(bounded_lengths) == 0:
            raise ValueError(f'{metadata_name} is empty?')

        fig, ax = plt.subplots()
        ax = sns.distplot(bounded_lengths, label=metadata_name)
        ax.legend()
        fig.savefig(f'graphs/speech_length/{metadata_name}.pdf')


def main() -> None:
    # Congressional Record
    post_WWII_sessions = range(79, 112)
    output_dir = '../../data/debug_partitioned_corpora'
    min_speech_length = 20
    num_chunks = 10
    input_encoding: str = 'mac_roman'  # NOTE

    os.makedirs(output_dir)
    print('Session\t\tSpeech Count\tWord Count\tShort Speech\tMissing Metadata')
    for session in post_WWII_sessions:
        original_corpus_path = f'../../data/raw/bound/speeches_{session:0>3d}.txt'
        metadata_path = f'../../data/raw/bound/{session:0>3d}_SpeakerMap.txt'
        stats = partition_corpus(
            session, original_corpus_path, metadata_path, output_dir,
            min_speech_length, num_chunks, input_encoding)
        speech_count, word_count, short_speech_count, missing_metadata_count = stats
        short_speech_ratio = short_speech_count / speech_count
        missing_metadata_ratio = missing_metadata_count / speech_count
        remaining_speech_count = speech_count - short_speech_count - missing_metadata_count
        print(f'{session}\t\t'
              f'{remaining_speech_count:,}\t\t'
              f'{word_count:,}\t'
              f'{short_speech_ratio:.2%}\t\t'
              f'{missing_metadata_ratio:.2%}')

    # # UCSB Presidency Project
    # partition_jsonl_corpus(
    #     corpus_path='../../data/raw/UCSB_presidency_project.jsonl',
    #     output_dir='../../data/interim/partitioned_presidency',
    #     num_chunks=10,
    #     input_encoding='utf_8')


if __name__ == '__main__':
    main()
