import pickle
import json
import re
import os
from collections import Counter
from typing import Set, Tuple, List, Dict, Iterable, Optional

import nltk
from nltk.collocations import (
    BigramCollocationFinder, BigramAssocMeasures,
    TrigramCollocationFinder, TrigramAssocMeasures)
from tqdm import tqdm


def load_parsing_result(
        parsing_result_dir: str,
        session: int,
        party: str,
        num_chunks: int,
        auto_caching: Optional[Set[str]],
        ) -> List[Dict]:
    """
    If auto_caching is True, it pickles the decoded JSON, so re-loading
    the same JSON will automatically load the pickled binary.

    Note that there is no mechainsm to check and update the cache file
    if the original file has been updated. In that case, you should manually
    delete the outdated cache files.
    """
    cache_path = os.path.join(parsing_result_dir, f'cache_{session}_{party}.pickle')
    if not os.path.exists(cache_path):  # haven't cached yet
        filtered_sentences: List[Dict] = []
        append_to_filtered_sentences = filtered_sentences.append
        for chunk in tqdm(range(num_chunks),
                          desc=f'Decoding parsing result of {session}_{party}'):
            in_path = os.path.join(
                parsing_result_dir, f'{session}_{party}{chunk}.json')
            with open(in_path) as parsing_result_file:
                parsing_result = json.load(parsing_result_file)
            for original_sentence in parsing_result['sentences']:
                append_to_filtered_sentences(
                    {key: original_sentence[key]
                     for key in original_sentence
                     if key in auto_caching})  # type: ignore

        if auto_caching:
            tqdm.write(f'Caching decoded json to {cache_path}... ')
            with open(cache_path, 'wb') as cache_file:
                pickle.dump(filtered_sentences, cache_file, protocol=-1)
        return filtered_sentences

    else:  # already cached
        tqdm.write(f'Loading {cache_path}...')
        with open(cache_path, 'rb') as cache_file:
            filtered_sentences = pickle.load(cache_file)
        return filtered_sentences


def sort_frequency_and_write(
        list_of_phrases: Iterable,
        out_path: str,
        min_frequency: int
        ) -> None:
    if isinstance(list_of_phrases, Counter):
        frequency = list_of_phrases
    else:
        frequency = Counter(list_of_phrases)
    output_iterable = [
        (frequency[phrase], phrase)
        for phrase in frequency
        if frequency[phrase] > min_frequency]
    output_iterable.sort(key=lambda t: t[0], reverse=True)
    with open(out_path, 'w') as out_file:
        for freq, phrase in output_iterable:
            out_file.write(f'{freq}\t{phrase}\n')

# TODO normalize NER results
def extract_named_entities(
        sentences: List[Dict],
        out_path: str,
        min_frequency: int
        ) -> None:
    contains_number = re.compile(r'\d')  # HACK check if still necessary with new NER settings
    entities = []
    for sent in sentences:
        if 'entitymentions' in sent:
            for entity in sent['entitymentions']:
                entity_text = entity['text']
                if (len(entity_text.split()) > 1  # Excluding Unigrams
                        and not contains_number.search(entity_text)):
                    entities.append(entity_text.lower())  # TODO remove punctuaion here?
    sort_frequency_and_write(entities, out_path, min_frequency)


def extract_noun_and_verb_phrases(
        sentences: List[Dict],
        out_path: str,
        discard_tokens: Set[str],
        stop_words: Set[str],
        min_frequency: int
        ) -> None:
    contains_number = re.compile(r'\d')  # HACK check if still necessary with new NER settings

    def normalize(iterable_of_tokens: Tuple[str, ...]) -> Tuple[str, ...]:
        """ remove stop_words, punctuation; all lowercase, """
        return tuple(word.lower()
                     for word in iterable_of_tokens
                     if word.lower() not in stop_words)

    def heuristic(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
        """ permits the second token to be a stop word """
        for token in tokens:
            if contains_number.search(token):
                return ('',)
        if len(tokens) == 2:
            return normalize(tokens)
        elif len(tokens) == 3:
            tokens = tuple(t.lower() for t in tokens)
            if (tokens[0] not in stop_words
                    and tokens[1] not in discard_tokens
                    and tokens[2] not in stop_words):
                return tokens
            else:
                return normalize(tokens)
        else:
            return ('',)

    string_to_tree = nltk.tree.Tree.fromstring
    parse_trees = (string_to_tree(s['parse']) for s in sentences)
    phrases_in_doc: List[str] = []
    for tree in parse_trees:
        phrases_in_sentence: Set[str] = set()
        for subtree in tree.subtrees():
            if subtree.label() != 'NP' and subtree.label() != 'VP':
                continue
            phrase = heuristic(subtree.leaves())
            if 1 < len(phrase) < 4:  # bigrams & trigrams only
                phrases_in_sentence.add(' '.join(phrase))
        phrases_in_doc.extend(list(phrases_in_sentence))
    sort_frequency_and_write(phrases_in_doc, out_path, min_frequency)


def compute_collocation(
        corpora_dir: str,
        session: int,
        party: str,
        num_chunks: int,
        bigram_out_path: str,
        trigram_out_path: str,
        discard_tokens: Set[str],
        stop_words: Set[str],
        min_frequency: int
        ) -> None:
    """
    discard_tokens should be a subset of stop_words. This is used for
    a heuristic to filter trigrams, where the second word is permitted
    to be a stop word (e.g. "freedom of speech") but not a discarded token
    (e.g. "I yield to"). The first and third words can never be a stop word.
    """
    tokenized_corpus: List[str] = []
    for chunk_index in range(num_chunks):
        corpus_path = os.path.join(corpora_dir, f'{session}_{party}{chunk_index}.txt')
        with open(corpus_path) as corpus_file:
            raw_text = corpus_file.read()
        tokens: List[str] = nltk.tokenize.word_tokenize(raw_text)
        tokens = [t.lower() for t in tokens
                  if t not in discard_tokens
                  and not t.isdigit()]
        tokenized_corpus.extend(tokens)
    del tokens

    bigram_finder = BigramCollocationFinder.from_words(tokenized_corpus)
    bigram_finder.apply_freq_filter(min_frequency)
    bigram_finder.apply_word_filter(
        lambda word: word in stop_words)
    bigrams = bigram_finder.score_ngrams(BigramAssocMeasures().raw_freq)

    trigram_finder = TrigramCollocationFinder.from_words(tokenized_corpus)
    trigram_finder.apply_freq_filter(min_frequency)
    trigram_finder.apply_ngram_filter(
        lambda w1, w2, w3:
        (w1 in stop_words) or (w3 in stop_words) or (w2 in discard_tokens))
    trigrams = trigram_finder.score_ngrams(TrigramAssocMeasures().raw_freq)

    num_tokens = len(tokenized_corpus)
    with open(bigram_out_path, 'w') as bigram_file:
        for bigram, relative_freq in bigrams:
            absolute_freq = relative_freq * num_tokens
            bigram_str = ' '.join(bigram)
            bigram_file.write(f'{absolute_freq:.0f}\t{bigram_str}\n')
    with open(trigram_out_path, 'w') as trigram_file:
        for trigram, relative_freq in trigrams:
            absolute_freq = relative_freq * num_tokens
            trigram_str = ' '.join(trigram)
            trigram_file.write(f'{absolute_freq:.0f}\t{trigram_str}\n')


def aggregate_phrases(
        Dem_phrase_sources: List[str],
        GOP_phrase_sources: List[str],
        out_path: str,
        top_k_phrases_per_source: int,
        min_frequency: int
        ) -> None:
    """load named_entities, NP_VP, collocations, write statistics"""

    def load_counters(counter_paths: List[str]) -> Counter:
        union_counter: Counter = Counter()
        for source_path in counter_paths:
            with open(source_path) as in_file:
                temp_counter: Counter = Counter()
                for line_num, line in enumerate(in_file):
                    if line_num > top_k_phrases_per_source:
                        break
                    frequency, phrase = line.split('\t')
                    frequency = int(frequency)
                    if frequency < min_frequency:
                        break
                    phrase = phrase.strip()
                    temp_counter[phrase] = frequency
            union_counter = union_counter | temp_counter
        return union_counter

    Dem_counter = load_counters(Dem_phrase_sources)
    GOP_counter = load_counters(GOP_phrase_sources)
    phrase_counter = Dem_counter + GOP_counter

    output_iterable = [
        (len(phrase.split()), Dem_counter[phrase], GOP_counter[phrase], phrase)
        for phrase in phrase_counter]

    # When replacing words with underscored phrases,
    # replace long phrases first, and then frequent phrases.
    output_iterable.sort(key=lambda t: (t[0], t[1]), reverse=True)
    with open(out_path, 'w') as out_file:
        for _, D_freq, R_freq, phrase in output_iterable:
            out_file.write(f'{D_freq}\t{R_freq}\t{phrase}\n')


def main() -> None:
    sessions = range(79, 112)
    parsing_result_dir = '../../data/interim/CoreNLP_parsed'
    num_parsing_chunks = 10
    out_dir = '../../data/interim/'

    # Pickle these from the parsing results
    auto_caching: Optional[Set[str]] = {'parse', 'entitymentions'}

    # For compute_collocation
    # Note partitioned corpora removed speeches without known speaker metadata
    corpora_dir = 'partitioned_corpora'

    # Noun phrases/ verb phrases including these words will be excluded
    procedural_words = {
        'yield', 'motion', 'order', 'ordered', 'quorum', 'roll', 'unanimous',
        'mr.', 'madam', 'speaker', 'chairman', 'president', 'senator',
        'gentleman', 'colleague', 'colleagues', '...', '``', "''", '--'}
    punctuations = set(char for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    discard_tokens = punctuations.union(procedural_words)
    stop_words = discard_tokens.union(
        set(nltk.corpus.stopwords.words('english')))

    # For aggregate_phrases
    min_frequency_per_source = 15
    top_k_phrases_per_source = 1000
    final_min_frequency = 30

    # Sanity check that parsing result files exist
    # sane = True
    # for session in sessions:
    #     for party in ('D', 'R'):
    #         for chunk in range(num_parsing_chunks):
    #             in_path = f'{parsing_result_dir}/{session}_{party}{chunk}.json'
    #             if not os.path.isfile(in_path):
    #                 print(f'{in_path} does not exist!')
    #                 sane = False
    # if not sane:
    #     raise FileNotFoundError()

    print(f'Processing sessions from {sessions}')
    for session in tqdm(sessions, desc='Sessions'):
        for party in ('D', 'R'):
            name_entities_path = os.path.join(
                out_dir, 'named_entities', f'{session}_{party}.txt')
            noun_and_verb_phrases_path = os.path.join(
                out_dir, 'noun_and_verb_phrases', f'{session}_{party}.txt')
            collocation_bigram_path = os.path.join(
                out_dir, 'collocation_bigram', f'{session}_{party}.txt')
            collocation_trigram_path = os.path.join(
                out_dir, 'collocation_trigram', f'{session}_{party}.txt')

            # sentences = load_parsing_result(
            #     parsing_result_dir, session, party, num_parsing_chunks, auto_caching)
            # extract_named_entities(
            #     sentences, name_entities_path, min_frequency_per_source)
            # extract_noun_and_verb_phrases(
            #     sentences, noun_and_verb_phrases_path,
            #     discard_tokens, stop_words, min_frequency_per_source)
            # del sentences

            # compute_collocation(
            #     corpora_dir, session, party, num_parsing_chunks,
            #     collocation_bigram_path, collocation_trigram_path,
            #     discard_tokens, stop_words, min_frequency_per_source)

        # Combine phrases from both parties
        Dem_phrase_sources = [
            os.path.join(out_dir, 'named_entities', f'{session}_D.txt'),
            os.path.join(out_dir, 'noun_and_verb_phrases', f'{session}_D.txt'),
            os.path.join(out_dir, 'collocation_bigram', f'{session}_D.txt'),
            os.path.join(out_dir, 'collocation_trigram', f'{session}_D.txt'),
            ]
        GOP_phrase_sources = [
            os.path.join(out_dir, 'named_entities', f'{session}_R.txt'),
            os.path.join(out_dir, 'noun_and_verb_phrases', f'{session}_R.txt'),
            os.path.join(out_dir, 'collocation_bigram', f'{session}_R.txt'),
            os.path.join(out_dir, 'collocation_trigram', f'{session}_R.txt')
        ]
        final_output_path = os.path.join(out_dir, 'aggregated_phrases', f'{session}.txt')
        aggregate_phrases(
            Dem_phrase_sources, GOP_phrase_sources, final_output_path,
            top_k_phrases_per_source, final_min_frequency)


if __name__ == '__main__':
    main()
