import multiprocessing as mp
import subprocess
from tqdm import tqdm

# session = int(sys.argv[1])
sessions = range(79, 83)
# sessions = range(83, 87)
# sessions = range(87, 92)
# sessions = range(92, 97)
# sessions = range(97, 102)
# sessions = range(102, 107)
# sessions = range(107, 112)
# corpora = [
#     f'partitioned_corpora/{session}_{party}{chunk}.txt'
#     for session in sessions
#     for party in ('D', 'R')
#     for chunk in range(0, 10)]
# print(f'Parsing congressional sessions {sessions}\n\n\n')

corpora = [
    f'../../data/interim/partitioned_presidency/{party}{chunk}.txt'
    for party in ('D', 'R')
    for chunk in range(0, 10)]

# Alternative, manually specify paths
# corpora = [
#     'partitioned_corpora/106_R9.txt',
#     'partitioned_corpora/108_D8.txt'
# ]

# Desktop setup
java_heap_memory = '15g'
num_cores = 4

# A computer cluster server with 256G RAM
# java_heap_memory = '20g'
# num_cores = 10

# Safer Choice
# java_heap_memory = '30g'
# num_cores = 8

output_dir = '../../data/interim/CoreNLP_presidency'
max_sentence_length = 100
CoreNLP_installation = '/data/nlp/resources/stanford-core-nlp/*'

def parse(path):
    parse_command = (
        'java '
        f'-Xmx{java_heap_memory} '
        f'-cp {CoreNLP_installation} '
        'edu.stanford.nlp.pipeline.StanfordCoreNLP '
        f'-file {path} '
        f'-outputDirectory {output_dir} '
        # '-annotators tokenize,ssplit,pos,lemma,ner,parse '
        '-annotators tokenize,ssplit,truecase,pos,lemma,ner,parse '
        '-truecase.overwriteText true '
        '-ner.applyNumericClassifiers false '
        '-ner.useSUTime false '
        # '-ner.combinationMode HIGH_RECALL '
        f'-parse.maxlen {max_sentence_length} '
        '-outputFormat json '
        '-replaceExtension')
    subprocess.run(parse_command.split())

with mp.Pool(num_cores) as team:
    _ = tuple(tqdm(team.imap_unordered(parse, corpora)))
