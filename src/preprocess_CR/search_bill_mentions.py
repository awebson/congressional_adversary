import re
import os
import csv
import sys
import pickle
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, NamedTuple, Counter, Optional

from tqdm import tqdm
import IPython


class Bill(NamedTuple):
    session: str
    number: str  # e.g. 'hrjres2'
    chamber: str
    topic: str
    summary: str
    first_line: str
    title: str  # informal title


class Speaker(NamedTuple):
    speaker_id: str
    last_name: str
    first_name: str
    chamber: str
    state: str
    gender: str
    party: str
    district: str


@dataclass
class Speech():
    speech_id: str
    date: str
    text: str
    speaker: Speaker
    mentions_bill: bool = False
    bill: Optional[Bill] = None


# global scope necessary for multiprocessing, sorry
in_path = '../../data/interim/titles_and_summaries_all_cleaned.txt'
out_dir = '../../data/interim/bill_mentions/daily'
os.makedirs(out_dir, exist_ok=True)
csv.field_size_limit(sys.maxsize)
with open(in_path) as file:
    reader = csv.reader(file, dialect=csv.excel_tab)
    bills = [Bill(*row) for row in reader]
title_to_bill: Dict[str, Bill] = {b.title: b for b in bills}
del bills


def export_bill_mentions(session: int) -> None:
    # load SpeakerMap files for speaker metadata
    speakers: Dict[str, Speaker] = {}
    speech_to_speaker: Dict[str, Speaker] = {}
    metadata_path = f'../../data/raw/daily/{session:0>3d}_SpeakerMap.txt'
    with open(metadata_path) as metadata_file:
        reader = csv.DictReader(metadata_file, delimiter='|')
        for row in reader:
            if row['nonvoting'] == 'nonvoting':
                continue
            if row['speakerid'] not in speakers:
                speakers[row['speakerid']] = Speaker(
                    row['speakerid'],
                    row['lastname'],
                    row['firstname'],
                    row['chamber'],
                    row['state'],
                    row['gender'],
                    row['party'],
                    row['district'])
            speech_to_speaker[row['speech_id']] = speakers[row['speakerid']]

    # load descr_ files for speech date information
    speech_to_date: Dict[str, str] = {}
    metadata_path = f'../../data/raw/daily/descr_{session:0>3d}.txt'
    with open(metadata_path) as metadata_file:
        reader = csv.DictReader(metadata_file, delimiter='|')
        for row in reader:
            speech_to_date[row['speech_id']] = row['date']

    # load speech files
    speeches: List[Speech] = []
    corpus_path = f'../../data/raw/daily/speeches_{session:0>3d}.txt'
    with open(corpus_path, encoding='mac_roman') as corpus_file:
        corpus_file.readline()  # discard header line
        for line in corpus_file:
            try:
                speech_id, speech_text = line.split('|')
            except ValueError:  # from spliting line with '|'
                continue
            # speech_count += 1
            if speech_id not in speech_to_speaker:
                # missing_metadata_count += 1
                continue
            # num_words = len(speech.split())
            # if num_words < min_speech_length:
            #     short_speech_count += 1
            #     continue
            # word_count += num_words
            speaker = speech_to_speaker[speech_id]
            date = speech_to_date[speech_id]
            speeches.append(
                Speech(speech_id, date, speech_text, speaker))

    queries = [
        b.title
        for b in title_to_bill.values()
        if int(b.session) == session]
    tqdm.write(f'{session}th Congress: {len(queries)} bills', end=', ')
    queries = map(re.escape, queries)  # type: ignore
    conglomerate_pattern = re.compile(f'({"|".join(queries)})')  # type: ignore

    num_mentions: Counter[str] = Counter()
    for speech in speeches:
        # line = line.lower()
        match = conglomerate_pattern.search(speech.text)
        if not match:
            continue
        bill_title = match.group(0)
        num_mentions[bill_title] += 1
        speech.mentions_bill = True
        speech.bill = title_to_bill[bill_title]
    tqdm.write(f'{sum(num_mentions.values())} matches')

    # IPython.embed()
    # sys.exit()
    out_path = os.path.join(out_dir, f'cache_{session}.pickle')
    with open(out_path, 'wb') as pickle_file:
        pickle.dump((speeches, num_mentions), pickle_file)


def main() -> None:
    scraped_sessions = list(range(97, 115))  # bound 93-115
    with mp.Pool(12) as team:
        _ = tuple(
            tqdm(team.imap_unordered(export_bill_mentions, scraped_sessions),
                 total=len(scraped_sessions)))


if __name__ == '__main__':
    main()
