import csv
from typing import Set, Tuple, NamedTuple, List, Dict, Counter, Optional

import torch
from torch import nn
import numpy as np
from scipy.stats import spearmanr

np.random.seed(42)
torch.manual_seed(42)


class Embedding():

    def __init__(
            self,
            path: str,
            source: Optional[str] = None,
            device: torch.device = torch.device('cuda')
            ) -> None:
        self.device = device
        if source == 'decomposer':
            self.init_from_decomposer(path)
        elif source == 'recomposer':
            self.init_from_recomposer(path)
        # elif source == 'tensorboard':
        #     self.init_from_tensorboard(path)
        # elif source == 'skip_gram':
        #     self.init_from_skip_gram(path)
        elif source == 'plain_text':
            self.init_from_plain_text(path)
        else:
            raise ValueError('Unknown embedding source.')

    def init_from_decomposer(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        model = payload['model']
        self.word_to_id = model.word_to_id
        self.id_to_word = model.id_to_word
        # self.Dem_frequency: Counter[str] = model.Dem_frequency
        # self.GOP_frequency: Counter[str] = model.GOP_frequency
        self.embedding = model.export_embedding(device=self.device)
        self.embedding.requires_grad = False

    def init_from_recomposer(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        model = payload['model']
        self.word_to_id = model.word_to_id
        self.id_to_word = model.id_to_word
        self.embedding = model.deno_decomposer.embedding.weight.detach()

    def init_from_plain_text(self, path: str) -> None:
        id_generator = 0
        word_to_id: Dict[str, int] = {}
        embeddings: List[List[float]] = []
        with open(path) as embedding_file:
            vocab_size, num_dimensions = map(int, embedding_file.readline().split())
            print(f'vocab_size = {vocab_size:,}, num_dimensions = {num_dimensions}')
            print(f'Loading embeddings from {path}', flush=True)
            for line in embedding_file:
                line: List[str] = line.split()  # type: ignore
                word = line[0]
                vector = list(map(float, line[-num_dimensions:]))
                embeddings.append(vector)
                word_to_id[word] = id_generator
                id_generator += 1
        print('Done')
        self.id_to_word = {val: key for key, val in word_to_id.items()}
        self.word_to_id = word_to_id
        self.embedding = torch.tensor(embeddings, device=self.device)

    def cosine_similarity(self, query1: str, query2: str) -> float:
        try:
            query1_id = self.word_to_id[query1]
        except KeyError as error:
            print(f'Out of vocabulary: {query1}')
            raise error
        try:
            query2_id = self.word_to_id[query2]
        except KeyError as error:
            print(f'Out of vocabulary: {query2}')
            raise error

        v1 = self.embedding[query1_id]
        v2 = self.embedding[query2_id]
        return nn.functional.cosine_similarity(v1, v2, dim=0).item()

    def nearest_neighbor(self, query: str, top_k: int = 10) -> None:
        query_id = self.word_to_id[query]
        query_vec = self.embedding[query_id]

        cos_sim = nn.functional.cosine_similarity(
            query_vec.view(1, -1), self.embedding)
        top_neighbor_ids = cos_sim.argsort(descending=True)[1:top_k + 1]
        for neighbor_id in top_neighbor_ids:
            neighbor_id = neighbor_id.item()
            neighbor_word = self.id_to_word[neighbor_id]
            sim = cos_sim[neighbor_id]
            print(f'{sim:.4f}\t{neighbor_word}')
        print('\n')


class PhrasePair(NamedTuple):
    query: str
    neighbor: str
    deno_sim: float
    cono_sim: float


def load_cherry(path, exclude_hard_examples=True):
    data = []
    with open(path) as file:
        if path.endswith('tsv'):
            reader = csv.DictReader(file, dialect=csv.excel_tab)
        else:
            reader = csv.DictReader(file)
        for row in reader:
            if row['semantic_similarity'] and row['cono_similarity']:
                if (exclude_hard_examples and
                        'hard example' in row['comment'].lower()):
                    continue
                data.append(PhrasePair(
                    row['query'],
                    row['neighbor'],
                    # row['query_words'],
                    # row['neighbor_words'],
                    float(row['semantic_similarity']),
                    float(row['cono_similarity'])))
    print(f'Loaded {len(data)} labeled entries at {path}')
    return data


def load_MTurk_result(path):
    data = []
    with open(path) as file:
        if path.endswith('tsv'):
            reader = csv.DictReader(file, dialect=csv.excel_tab)
        else:
            reader = csv.DictReader(file)
        for row in reader:
            qc = row['median_query_cono']
            nc = row['median_neighbor_cono']
            if qc and nc:  # nonempty string
                qc = float(qc)
                nc = float(nc)
                if qc == 0 or nc == 0:  # unable to judge
                    continue

                # cono_sim = 5 - abs(qc - nc)

                if ((qc > 3 and nc > 3)
                        or (qc < 3 and nc < 3)
                        or (qc == 3 and nc == 3)):
                    cono_sim = 5
                else:
                    cono_sim = 1

                data.append(PhrasePair(
                    row['query_words'],
                    row['neighbor_words'],
                    float(row['median_deno']),
                    cono_sim))
    print(f'Loaded {len(data)} labeled entries at {path}')
    return data


def correlate_sim_deltas(model, ref_model, phrase_pairs, verbose=False):
    label_deltas = []
    model_deltas = []
    if verbose:
        print(f'deno_sim\tcono_sim\tref_sim\tmodel_sim')

    for pair in phrase_pairs:
        try:
            sim = model.cosine_similarity(pair.query, pair.neighbor)
            ref_sim = ref_model.cosine_similarity(pair.query, pair.neighbor)
        except KeyError:
            continue
        model_delta = sim - ref_sim
        model_deltas.append(model_delta)
        label_deltas.append(pair.deno_sim - pair.cono_sim)

        if verbose:
            print(f'{pair.deno_sim}  {pair.cono_sim}  {ref_sim:.2%}  {sim:.2%}  '
                  f'{pair.query}  {pair.neighbor}')

    median = np.median(model_deltas)
    mean = np.mean(model_deltas)
    stddev = np.std(model_deltas)
    rho, _ = spearmanr(model_deltas, label_deltas)
    return rho, median, mean, stddev


def preview(things):
    for stuff in things:
        q, n, d, c = stuff
        print(d, c, q, n, sep='\t')


def same_deno(pair):
    return pair.deno_sim >= 3


def same_cono(pair):
    return pair.cono_sim >= 3


def is_euphemism(pair) -> bool:
    return same_deno(pair) and not same_cono(pair)


def is_party_platform(pair) -> bool:
    return not same_deno(pair) and same_cono(pair)


cherry_pairs = [
    # Luntz Report, all GOP euphemisms
    ('government', 'washington'),
    # ('private_account', 'personal_account'),
    # ('tax_reform', 'tax_simplification'),
    ('estate_tax', 'death_tax'),
    ('capitalism', 'free_market'),  # global economy, globalization
    # ('outsourcing', 'innovation'),  # "root cause" of outsourcing, regulation
    ('undocumented', 'illegal_aliens'),  # OOV undocumented_workers
    ('foreign_trade', 'international_trade'),  # foreign, global all bad
    # ('drilling_for_oil', 'exploring_for_energy'),
    # ('drilling', 'energy_exploration'),
    # ('tort_reform', 'lawsuit_abuse_reform'),
    # ('trial_lawyer', 'personal_injury_lawyer'),  # aka ambulance chasers
    # ('corporate_transparency', 'corporate_accountability'),
    # ('school_choice', 'parental_choice'),  # equal_opportunity_in_education
    #('healthcare_choice', 'right_to_choose')

    # Own Cherries
    ('public_option', 'governmentrun'),
    ('political_speech', 'campaign_spending'),  # hard example
    ('cut_taxes', 'trickledown')  # OOV supplyside
]


cherry_words = (
    'military_budget', 'defense_budget',
    # 'nuclear_option', 'constitutional_option',
    'prochoice',  # 'proabortion',
    'star_wars',  # 'strategic_defense_initiative',
    'political_speech', 'campaign_spending',
    'singlepayer',  # 'socialized_medicine',
    # 'voodoo', 'supplyside',
    'tax_expenditures',  # 'spending_programs',
    'waterboarding', 'interrogation',
    'cap_and_trade', 'national_energy_tax',
    'governmentrun', 'public_option',
    'medical_liability_reform',  # 'tort_reform',
    # 'corporate_profits', 'earnings',
    # 'equal_pay',  # 'the_paycheck_fairness_act',
    'military_spending',  # 'washington_spending',
    'higher_taxes',  # 'bigger_government',
    'social_justice',  # 'womens_rights',
    # 'national_health_insurance', # 'welfare_state',
    'nuclear_war', 'deterrence',
    'suffrage',  # 'womens_rights',
    'inequality', 'racism',
    # 'sweatshops', 'factories',
    'trickledown', 'cut_taxes',
    'equal_pay', 'pay_discrimination',
    'wealthiest_americans', 'tax_breaks',
    'record_profits', 'big_oil_companies',
    # 'private_insurance_companies', 'medicare_advantage_program',
    # 'trickledown',  # 'universal_health_care',
    'big_banks',  # 'occupation_of_iraq',
    # 'obamacare', 'islamists'
)

generic_words = (
    'government',
    'taxes',
    'laws',
    'jobs',
    'tariff',
    'health_care',
    'finance',
    'social_security',
    'medicare',
    'regulations',
    'immigration',
    'research',
    'technology',
)
