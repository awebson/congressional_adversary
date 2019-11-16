import re
import csv
import random
from typing import Set, List, Dict

from tqdm import tqdm

random.seed(42)

class LabeledPhrasePair():

    def __init__(self, data: Dict):
        self.comment = data['comment'].lower()
        self.cherry = 'cherry' in self.comment
        self.hard_example = 'hard_example' in self.comment

        if data['semantic_similarity']:
            self.deno_sim = int(data['semantic_similarity'])
        if data['cono_similarity']:
            self.cono_sim = int(data['cono_similarity'])

        # # Low Recall, High Precision
        # if hasattr(self, 'deno_sim') or data['evaluable'].upper() == 'T':
        #     self.evaluable = True
        # else:
        #     self.evaluable = False

        # High Recall, Low Precision
        if data['evaluable'].upper() == 'F':
            self.evaluable = False
        else:
            self.evaluable = True

        self.query = data['query']
        self.q_D_freq = int(data['query_D_freq'])
        self.q_R_freq = int(data['query_R_freq'])
        self.q_R_ratio = str(data['query_R_ratio'])

        self.neighbor = data['neighbor']
        self.n_D_freq = int(data['neighbor_D_freq'])
        self.n_R_freq = int(data['neighbor_R_freq'])
        self.n_R_ratio = str(data['neighbor_R_ratio'])

        self.cosine_sim = float(data['cosine_similarity'])


def main() -> None:
    phrase_path = '../../data/evaluation/labeled_Dem_samples.tsv'
    corpus_path = '../../data/processed/plain_text/for_real/corpus.txt'
    out_path = '../../data/evaluation/MTurk_dumb.csv'
    contexts_per_phrase = 5

    corpus: List[str] = []
    with open(corpus_path) as file:
        for line in file:
            corpus.append(line)
    random.shuffle(corpus)

    phrases: List[LabeledPhrasePair] = []
    with open(phrase_path) as file:
        reader = csv.DictReader(file, dialect=csv.excel_tab)
        for raw_data in reader:
            try:
                phrases.append(LabeledPhrasePair(raw_data))
            except ValueError:  # empty rows
                continue

    evaluable_queries: Set[str] = {ph.query for ph in phrases if ph.evaluable}
    phrases = [ph for ph in phrases if ph.query in evaluable_queries]

    # import IPython
    # IPython.embed()

    column_names = list(vars(phrases[0]).keys())
    column_names += [f'context{i}' for i in range(contexts_per_phrase)]
    out_file = open(out_path, 'w')
    csv_writer = csv.DictWriter(out_file, column_names)
    csv_writer.writeheader()

    def code_injection(regex_match) -> str:
        return '<b>' + regex_match.group() + '</b>'

    for pair in tqdm(phrases):
        context_i = 0
        phrase_pattern = re.compile(pair.query)
        context_pattern = re.compile(
            r'(\w*\s){1,30}' + pair.query + r'(\s\w*){1,30}')
        for speech in corpus:
            if pair.query not in speech:
                continue
            context = context_pattern.search(speech)
            if context:
                context = context.group().strip()
                context = phrase_pattern.sub(code_injection, speech)
                setattr(pair, f'context{context_i}', context)
                context_i += 1
            if context_i >= contexts_per_phrase:
                break

        csv_writer.writerow(vars(pair))
    out_file.close()


if __name__ == '__main__':
    main()
