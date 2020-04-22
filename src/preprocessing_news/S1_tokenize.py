import xml.etree.ElementTree as ET
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Iterable, Optional

import stanza
from tqdm import tqdm

# Sentence = stanza.models.common.doc.Sentence
# Token = stanza.models.common.doc.Token
# Span = stanza.models.common.doc.Span

@dataclass
class Sentence():
    tokens: List[List[str]]
    underscored_tokens: List[List[str]]
    entities: List[Tuple[str, str]]  # ent.type, ent.text
    normalized_tokens: List[List[str]] = field(default_factory=list)
    subsampled_tokens: List[List[str]] = field(default_factory=list)
    numerical_tokens: List[List[int]] = field(default_factory=list)


@dataclass
class LabeledDoc():
    uid: str
    title: str
    url: str
    party: str  # left, left-center, least, right-center, right
    partisan: bool
    text: str
    date: Optional[str] = None
    sentences: Optional[List[Sentence]] = None


EXCLUDE = {'DATE', 'ORDINAL', 'CARDINAL', 'PERCENT', 'MONEY', 'TIME', 'QUANTITY'}

def underscore_NER(tokens):
    new_seq = []
    for index, token in enumerate(tokens):
        if token.ner == 'O':  # No NER
            new_seq.append(token.text)
            continue
        prefix, ent_type = token.ner.split('-')
        if ent_type in EXCLUDE:
            new_seq.append('<NUM>')
        elif prefix == 'S':  # single-token length
            new_seq.append(token.text)

        elif prefix == 'B':  # multi-token phrase
            joint_token = [token.text, ]
            for token_ahead in tokens[index + 1:]:
                if token_ahead.ner == 'O':
                    break
                prefix_ahead, _ = token_ahead.ner.split('-')
                if prefix_ahead == 'I':
                    joint_token.append(token_ahead.text)
                elif prefix_ahead == 'E':
                    joint_token.append(token_ahead.text)
                    break
                elif prefix_ahead == 'B' or prefix_ahead == 'S':
                    break
                else:
                    print(token, token.ner)
                    print(token_ahead, token.ner)
                    print([(t.text, t.ner) for t in tokens])
                    raise RuntimeError('Malformed NER IOB tag')
            new_seq.append('_'.join(joint_token))
        else:
            continue
    return new_seq


def partition(corpus: List, num_chunks: int) -> Iterable[List]:
    chunk_size = len(corpus) // num_chunks
    corpus_index = 0
    chunk_index = 0
    while chunk_index < num_chunks - 1:
        yield corpus[corpus_index:corpus_index + chunk_size]
        corpus_index += chunk_size
        chunk_index += 1
    yield corpus[corpus_index:-1]


def main() -> None:
    print('Loading XML...')
    in_dir = Path.home() / 'Research/hyperpartisan_news'
    # metadata = ET.parse(in_dir / 'ground-truth-training-bypublisher-20181122.xml')
    # corpus = ET.parse(in_dir / 'articles-training-bypublisher-20181122.xml')
    corpus = ET.parse(in_dir / 'articles-validation-bypublisher-20181122.xml')
    metadata = ET.parse(in_dir / 'ground-truth-validation-bypublisher-20181122.xml')

    labels = {
        stuff.attrib['id']: stuff.attrib
        for stuff in metadata.getroot()}
    print(f'Number of metadata entries = {len(labels):,}')

    print('Parsing XML...')
    data = []
    for article in corpus.getroot():
        attr = article.attrib
        text = []
        for para in article.itertext():  # pseudo-paragraph
            para = para.strip()
            if para:  # not whitespace
                text.append(para)
        text = ' '.join(text)

        if text:  # nonempty
            label = labels[attr['id']]
            data.append(LabeledDoc(
                uid=attr['id'],
                party=label['bias'],
                partisan=bool(label['hyperpartisan']),
                url=label['url'],
                title=attr['title'],
                date=attr.get('published-at', None),
                text=text))
        else:
            print('Missing:', article.attrib)
    print(f'Number of nonempty articles = {len(data):,}')

    stanza_processor = stanza.Pipeline(
        lang='en', processors='tokenize,ner',
        tokenize_batch_size=32, ner_batch_size=32)
    # stanza_processor = stanza.Pipeline(
    #   lang='en', processors={'tokenize': 'spacy'})


    # NOTE further dividing data for distributing among GPUs
    # data = data[:len(data) // 2]  # first half
    # data = data[len(data) // 2:]  # second half

    out_dir = Path('../../data/interim/news/validation')
    Path.mkdir(out_dir, parents=True, exist_ok=True)

    num_chunks = 100
    for part_index, some_docs in tqdm(
            enumerate(partition(data, num_chunks)),
            total=num_chunks,
            desc='Total'):

        # for doc in corpus:
        #     doc.compressed = [
        #         Sentence(underscore_NER(stanza_sent.tokens))
        #         for stanza_sent in doc.sentences]
        for doc in tqdm(some_docs, desc='Chunk'):
            processed = stanza_processor(doc.text)
            doc.sentences = [  # throw away extra info to conserve disk space
                Sentence(
                    tokens=[token.text for token in stanza_sent.tokens],
                    underscored_tokens=underscore_NER(stanza_sent.tokens),
                    entities=[(ent.type, ent.text) for ent in stanza_sent.ents])
                for stanza_sent in processed.sentences
            ]
        with open(out_dir / f'tokenized_{part_index}.pickle', 'wb') as file:
            pickle.dump(some_docs, file, protocol=-1)


if __name__ == '__main__':
    main()
