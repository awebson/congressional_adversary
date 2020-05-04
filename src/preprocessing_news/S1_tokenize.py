import xml.etree.ElementTree as ET
import pickle
from pathlib import Path
from typing import Set, List, Iterable

import stanza
from tqdm import tqdm

from data import LabeledDoc, Sentence


def parse_xml(
        corpus_path: Path,
        metadata_path: Path,
        ) -> List[LabeledDoc]:
    print('Loading XML...')
    metadata = ET.parse(metadata_path)  # type: ignore
    corpus = ET.parse(corpus_path)  # type: ignore

    labels = {
        stuff.attrib['id']: stuff.attrib
        for stuff in metadata.getroot()}
    print(f'Number of metadata entries = {len(labels):,}')

    print('Parsing XML...')
    data: List[LabeledDoc] = []
    existed: Set[str] = set()
    duplicate_count = 0
    for article in corpus.getroot():
        attr = article.attrib
        text = []
        # Joining the body text of the document.
        for para in article.itertext():  # pseudo-paragraph
            para = para.strip()
            if para:  # not whitespace
                text.append(para)
        text = ' '.join(text)
        if text in existed:  # NOTE better deduplication to come later. Sorry!
            duplicate_count += 1
            continue
        else:
            existed.add(text)

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
    print(f'Number of duplicated articles = {duplicate_count:,}')
    print(f'Number of nonduplicate articles = {len(data):,}')
    return data


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
    in_dir = Path.home() / 'Research/hyperpartisan_news'
    out_dir = Path('../../data/interim/news/train')
    Path.mkdir(out_dir, parents=True, exist_ok=True)
    corpus = in_dir / 'articles-training-bypublisher-20181122.xml'
    metadata = in_dir / 'ground-truth-training-bypublisher-20181122.xml'
    # dev_corpus = in_dir / 'articles-validation-bypublisher-20181122.xml'
    # dev_metadata = in_dir / 'ground-truth-validation-bypublisher-20181122.xml'
    data = parse_xml(corpus, metadata)

    processor = stanza.Pipeline(
        lang='en', processors='tokenize', tokenize_batch_size=4096)
    # processor = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'})

    for part_index, some_docs in tqdm(
            enumerate(partition(data, 100)), total=100, desc='Total'):
        for doc in tqdm(some_docs, desc='Chunk'):
            processed = processor(doc.text)
            doc.sentences = [
                Sentence([token.text for token in stanza_sent.tokens])
                for stanza_sent in processed.sentences]
        with open(out_dir / f'tokenized_{part_index}.pickle', 'wb') as file:
            pickle.dump(some_docs, file, protocol=-1)


if __name__ == '__main__':
    main()
