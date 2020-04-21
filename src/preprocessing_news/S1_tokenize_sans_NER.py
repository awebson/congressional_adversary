import xml.etree.ElementTree as ET
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Iterable, Optional

import stanza
from tqdm import tqdm

# Sentence = stanza.models.common.doc.Sentence
# Token = stanza.models.common.doc.Token
# Span = stanza.models.common.doc.Span

@dataclass
class Sentence():
    tokens: List[List[str]]
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
#     entities: Optional[List[Span]] = None


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
    metadata = ET.parse(in_dir / 'ground-truth-training-bypublisher-20181122.xml')
    corpus = ET.parse(in_dir / 'articles-training-bypublisher-20181122.xml')

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


    processor = stanza.Pipeline(
        lang='en', processors='tokenize', tokenize_batch_size=20_000)
    # processor = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'})

    out_dir = Path('../../data/interim/news/')
    Path.mkdir(out_dir, parents=True, exist_ok=True)

    for part_index, some_docs in tqdm(
            enumerate(partition(data, 60)), total=60, desc='Total'):
        for doc in tqdm(some_docs, desc='Chunk'):
            processed = processor(doc.text)
            doc.sentences = [
                Sentence([token.text for token in stanza_sent.tokens])
                for stanza_sent in processed.sentences]
        with open(out_dir / f'tokenized_{part_index}.pickle', 'wb') as file:
            pickle.dump(some_docs, file, protocol=-1)


if __name__ == '__main__':
    main()
