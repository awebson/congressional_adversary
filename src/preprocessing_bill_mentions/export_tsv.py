import pickle
import csv
import multiprocessing as mp
from copy import copy
from typing import List, Iterable, Dict

from tqdm import tqdm

from search_bill_mentions import Bill, Speaker, Speech

MIN_NUM_MENTIONS = 2
FIXED_SENT_LEN = 15
MIN_SENT_LEN = 5
base_dir = '../../data/interim/bill_mentions/'
print('Minimum number of mentions per bill =', MIN_NUM_MENTIONS)


punctuations = '!"#$%&\'()*+,-—./:;<=>?@[\\]^`{|}~'  # excluding underscore
remove_punctutation = str.maketrans('', '', punctuations)
remove_numbers = str.maketrans('', '', '0123456789')

from nltk.corpus import stopwords
procedural_words = {
    'yield', 'motion', 'order', 'ordered', 'quorum', 'roll', 'unanimous',
    'mr', 'madam', 'speaker', 'chairman', 'president', 'senator',
    'gentleman', 'colleague'}
stopwords = set(stopwords.words('english')).union(procedural_words)


fieldnames = (
    # Bill
    'session',
    'title',    # bill informal title
    'number',  # e.g. 'hrjres2'
    'topic',
    'summary',
    'first_line',

    # Speaker
    'speaker_id',
    'party',
    'last_name',
    'first_name',
    'chamber',
    'state',
    'gender',
    'district',

    # Speech
    'date',
    'speech_id',
    'cherry_mention',
    'faux_sentence'
)

# Load cherries
from evaluations import intrinsic_eval
Dem_pairs = intrinsic_eval.load_cherry(
    '../../data/evaluation/cherries/labeled_Dem_samples.tsv',
    exclude_hard_examples=True)
GOP_pairs = intrinsic_eval.load_cherry(
    '../../data/evaluation/cherries/labeled_GOP_samples.tsv',
    exclude_hard_examples=True)
val_data = Dem_pairs + GOP_pairs

euphemism = list(
    filter(intrinsic_eval.is_euphemism, val_data))
party_platform = list(
    filter(intrinsic_eval.is_party_platform, val_data))
party_platform += intrinsic_eval.load_cherry(
    '../../data/evaluation/cherries/remove_deno.tsv',
    exclude_hard_examples=False)

find = [pair.query for pair in euphemism]
find += [pair.neighbor for pair in euphemism]
find += [pair.query for pair in party_platform]
find += [pair.neighbor for pair in party_platform]
find = [s.replace('_', ' ') for s in find]
find = set(find)  # deduplicate
find.remove('earnings')  # manual
cherries = tuple(find)  # enforce iter order


# # Export TSV files without tokenize sentences
# cherry_stat = []
# base_dir = '../../data/interim/bill_mentions/'
# for session in range(93, 112):
#     in_path = base_dir + f'cache_{session}.pickle'
#     with open(in_path, 'rb') as in_file:
#         data: Dict[Bill, List[Speech]] = pickle.load(in_file)

#     out_path = base_dir + f'mentions_{session}.tsv'
#     out_file = open(out_path, 'w')
#     writer = csv.DictWriter(out_file, fieldnames, dialect=csv.excel_tab)
#     writer.writeheader()
#     num_mentions = 0
#     cherry_count = 0
#     for bill, mentions in data.items():
#         if len(mentions) < MIN_NUM_MENTIONS:
#             continue
#         bill = bill._asdict()
#         del bill['chamber']  # duplicate with speaker chamber

#         for speech in mentions:
#             output = copy(bill)
#             output.update(speech.speaker._asdict())
#             output['date'] = speech.date
#             output['text'] = speech.text

#             cherry_mentions = []
#             for i, context in enumerate(speech.contexts):
#                 output[f'context{i}'] = context
#                 for cherry in cherries:
#                     if cherry in context:
#                         cherry_mentions.append(cherry)
#             output['cherry_mentions'] = cherry_mentions
#             cherry_stat.append(len(cherry_mentions))

#             if len(cherry_mentions) > 0:
#                 cherry_count += 1

#             writer.writerow(output)
#             num_mentions += 1
#     out_file.close()
#     print(f'Session {session}, number of mentions = {num_mentions}')
#     # print(cherry_count / num_mentions)


def faux_sent_tokenize(line: str) -> Iterable[List[str]]:
    """discard procedural words and punctuations"""
    line = line.translate(remove_numbers).translate(remove_punctutation)
    words = [w for w in line.lower().split() if w not in stopwords]
    start_index = 0
    while (start_index + FIXED_SENT_LEN) < (len(words) - 1):
        yield words[start_index:start_index + FIXED_SENT_LEN]
        start_index += FIXED_SENT_LEN

    trailing_words = words[start_index:-1]
    if len(trailing_words) >= MIN_SENT_LEN:
        yield trailing_words


def export_tsv(session: int) -> None:
    in_path = base_dir + f'cache_{session}.pickle'
    with open(in_path, 'rb') as in_file:
        speeches, num_mentions = pickle.load(in_file)

    # Mark context speeches with bill denotation
    for speech_index, speech in enumerate(speeches):
        if speech.mentions_bill is False:
            continue
        if num_mentions[speech.bill.title] < MIN_NUM_MENTIONS:
            continue

        # NOTE hardcoded context_size
        for i in range(speech_index - 2, speech_index + 8):
            try:
                speeches[i].bill = speech.bill
            except IndexError:
                continue

    # Write to file
    out_path = base_dir + f'mentions_{session}.tsv'
    out_file = open(out_path, 'w')
    writer = csv.DictWriter(out_file, fieldnames, dialect=csv.excel_tab)
    writer.writeheader()
    num_sentences = 0
    for speech in speeches:
        if speech.bill is None:
            continue
        if num_mentions[speech.bill.title] < MIN_NUM_MENTIONS:
            continue

        speech_metadata = speech.speaker._asdict()
        deno = speech.bill._asdict()
        del deno['chamber']  # duplicate with speaker chamber

        # NOTE temporary: reduce file size
        del deno['summary']
        del deno['first_line']

        speech_metadata.update(deno)
        speech_metadata['date'] = speech.date
        speech_metadata['speech_id'] = speech.speech_id

        for sent in faux_sent_tokenize(speech.text):
            output = copy(speech_metadata)
            output['faux_sentence'] = ' '.join(sent)

            cherry_mentions = []
            for cherry in cherries:
                if cherry in sent:
                    cherry_mentions.append(cherry)
            if len(cherry_mentions) > 0:
                output['cherry_mention'] = cherry_mentions
                if len(cherry_mentions) > 1:
                    print(cherry_mentions)

            writer.writerow(output)
            num_sentences += 1

    tqdm.write(f'Session {session}: {num_sentences:,} faux sentences')
    out_file.close()


def main() -> None:
    # Export TSV with chopped up sentences & contexts, à la real training data
    sessions = range(93, 112)  # 93
    with mp.Pool(12) as team:
        _ = tuple(
            tqdm(team.imap_unordered(export_tsv, sessions),
                 total=len(sessions)))


if __name__ == '__main__':
    main()
