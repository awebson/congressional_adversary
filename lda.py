import pickle
import os
import random

from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath

# Constants
saved_lda = "ldamodel.bin"
num_topics = 42
desc_words_per_topic = 9

def display_topic(lda, common_dict, topic_no):
    term_list = lda.get_topic_terms(topic_no, topn=desc_words_per_topic)
    print("Word list for topic", topic_no)
    for word_id, val in term_list:
        print(common_dict[word_id], val)

def get_pn_corpus():

    pick_file = '../my_data/pn_underscored.pickle'
    vocab_file = "../newcong_ad_data/data/processed/bill_mentions/train_data.pickle"

    print("Loading pickle file...", pick_file)
    with open(pick_file, 'rb') as pickf:
        pickdata = pickle.load(pickf)

    print("Loading vocab file...", vocab_file)
    with open(vocab_file, 'rb') as vocabf:
        vocabdata = pickle.load(vocabf)
    word_to_id = vocabdata['word_to_id']

    print("Building documents...")
    pn_corpus = []
    for rec in pickdata:
        document = []
        for s in rec['under_sent']:
            #document.extend(s)
            document.extend(filter(lambda w: w in word_to_id, s))
        pn_corpus.append(document)
    
    return pn_corpus, word_to_id, pickdata

text_only, word_to_id, pickdata = get_pn_corpus()

# Create a corpus from a list of texts
print("Creating dictionary...")
common_dictionary = Dictionary(text_only)
common_corpus = [common_dictionary.doc2bow(text) for text in text_only]

# Train the model on the corpus.
if os.path.exists(saved_lda):
    print("Loading LDA...")
    lda = LdaModel.load(saved_lda)
else:
    print("Training LDA...")
    lda = LdaModel(common_corpus, num_topics=num_topics)
    print("Saving LDA...")
    lda.save(saved_lda)


# Pick several documents at random
num_documents = 8
doc_ids = random.sample(range(len(common_corpus)), num_documents)
for rndm_id in doc_ids:
    topic_list = lda.get_document_topics(common_corpus[rndm_id])
    title = pickdata[rndm_id]['title']
    print("Topic list for title:")
    print(title)
    print("")
    prob_topic_list = []
    for topic_no, prob in topic_list:
        prob_topic_list.append((prob, topic_no))
    prob_topic_list.sort(reverse=True)
    for prob, topic_no in prob_topic_list:
        print("Topic no", topic_no, "prob", prob)
        display_topic(lda, common_dictionary, topic_no)
        print("")

    