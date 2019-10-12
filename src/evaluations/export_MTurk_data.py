import re
import csv
import random
from typing import List, Dict

from tqdm import tqdm

random.seed(42)

class PhrasePair():

    def __init__(self, data: Dict):
        self.comment = data['comment'].lower()
        self.cherry = 'cherry' in self.comment
        self.hard_example = 'hard_example' in self.comment

        if data['semantic_similarity']:
            self.deno_sim = int(data['semantic_similarity'])
        if data['cono_similarity']:
            self.cono_sim = int(data['cono_similarity'])
        if (hasattr(self, 'deno_sim')
                or data['evaluable'] == 'T'
                or data['evaluable'] == 't'):
            self.evaluable = True
        else:
            self.evaluable = False

        self.query = data['query']
        self.q_D_freq = int(data['query_D_freq'])
        self.q_R_freq = int(data['query_R_freq'])
        self.q_R_ratio = str(data['query_R_ratio'])

        self.neighbor = data['neighbor']
        self.n_D_freq = int(data['neighbor_D_freq'])
        self.n_R_freq = int(data['neighbor_R_freq'])
        self.n_R_ratio = str(data['neighbor_R_ratio'])

        self.cosine_sim = float(data['cosine_similarity'])
        self.short_contexts: List[str] = []
        self.long_contexts: List[str] = []

    def export_dict(self) -> Dict:
        attrs = vars(self)

        for i, ctx in enumerate(self.short_contexts):
            column_name = f's_ctx{i}'
            attrs[column_name] = ctx

        for i, ctx in enumerate(self.long_contexts):
            column_name = f'l_ctx{i}'
            attrs[column_name] = ctx

        del attrs['short_contexts']
        del attrs['long_contexts']
        return attrs


def main() -> None:
    phrase_path = '../../data/evaluation/labeled_GOP_samples.tsv'
    corpus_path = '../../data/processed/plain_text/for_real/corpus.txt'
    out_path = '../../data/evaluation/MTurk.csv'

    corpus: List[str] = []
    with open(corpus_path) as file:
        for line in file:
            corpus.append(line)
    random.shuffle(corpus)

    phrases: List[PhrasePair] = []
    with open(phrase_path) as file:
        reader = csv.DictReader(file, dialect=csv.excel_tab)
        for raw_data in reader:
            try:
                phrases.append(PhrasePair(raw_data))
            except ValueError:  # empty rows
                continue

    phrases = [ph for ph in phrases if ph.evaluable]

    column_names = list(vars(phrases[0]).keys())
    column_names += [f's_ctx{i}' for i in range(10)]
    column_names += [f'l_ctx{i}' for i in range(10)]
    out_file = open(out_path, 'w')
    csv_writer = csv.DictWriter(out_file, column_names)
    csv_writer.writeheader()
    for pair in tqdm(phrases):
        short_pattern = re.compile(r'(\w*\s){1,8}' + pair.query + r'(\s\w*){1,8}')
        long_pattern = re.compile(r'(\w*\s){1,25}' + pair.query + r'(\s\w*){1,25}')

        for speech in corpus:
            if pair.query not in speech:
                continue
            # for i, context in enumerate(long_pattern.finditer(speech)):
            #     pair.long_contexts.append(context.group().strip())
            #     if i > 2:  # at most 2 samples from a single speech
            #         break
            long_context = long_pattern.search(speech)
            if long_context:
                pair.long_contexts.append(long_context.group().strip())
            if len(pair.long_contexts) > 5:
                break

        pair.short_contexts = [
            short_pattern.search(ctx).group()  # type: ignore
            for ctx in pair.long_contexts[:3]]

        csv_writer.writerow(pair.export_dict())

    out_file.close()


if __name__ == '__main__':
    main()

    # import cProfile
    # cProfile.run('main()', sort='cumulative')
