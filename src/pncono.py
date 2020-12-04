import pickle

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import random

from utils.experiment import Experiment
from ultradense import UltraDense, distro_thresh, preds_from_vector
from classifier import Classifier
 
# Experiment control knobs
experiment_name = "dense" # One of: cono, pos, sort, pairs, dense
use_avg_prec = False
binarize_embeddings = False # Implies !transform_embeddings
transform_embeddings = False
randomize_embeddings = False
use_pca = False # Otherwise, ICA. Only applies if transform_embeddings is True.
pn_corpus = False # Partisan News if True otherwise, Congressional Record
new_cr_corpus = True # Only applies if pn_corpus is false
use_saved_wordinfo = False

albert_pick_file = "../albert_wordlist.pickle"

with open(albert_pick_file, 'rb') as albert_file:
    albert_pick = pickle.load(albert_file)

saved_pickle_name = "intermediate.pickle"
if pn_corpus:
    pickle_file = "../data/ready/PN_proxy/train.pickle"
    embedding_file = "../data/pretrained_word2vec/PN_heading_proxy.txt"

else:

    #embedding_file = "/Users/tberckma/Documents/Brown/Research/AlbertProject/newcong_ad_data/data/pretrained_word2vec/for_real_SGNS_method_B.txt"
    #pickle_file = "/Users/tberckma/Documents/Brown/Research/AlbertProject/congressional_adversary/data/ready/CR_proxy/train.pickle"

    #pickle_file = "../../newcong_ad_data/data/processed/bill_mentions/train_data.pickle"
    #embedding_file = "../data/pretrained_word2vec/CR_proxy.txt"    

    if new_cr_corpus:
        pickle_file = "../data/ready/CR_topic_context3/train_data.pickle"
        #pickle_file = "../../newcong_ad_data/data/processed/bill_mentions/train_data.pickle"
        embedding_file = "../data/pretrained_word2vec/CR_bill_topic_context3.txt"    
    else:
        pickle_file = "../../newcong_ad_data/data/processed/bill_mentions/train_data.pickle"
        embedding_file = "/Users/tberckma/Documents/Brown/Research/AlbertProject/newcong_ad_data/data/pretrained_word2vec/for_real_SGNS.txt"

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
    # ('healthcare_choice', 'right_to_choose')

    # Own Cherries
    ('public_option', 'governmentrun'),
    ('political_speech', 'campaign_spending'),  # hard example
    ('cut_taxes', 'trickledown')  # OOV supplyside
]

if pn_corpus:
    grounding_name = 'ground'
else:
    if new_cr_corpus:
        grounding_name = 'ground'
    else:
        grounding_name = 'grounding'

deno_topic_table = {}

if pn_corpus:
    def calculate_cono_raw(grounding, word):
        """Find the connotation score between 0.0 and 4.0 for this word
    
        Used for Partisan News corpus.
    
        4.0 = Right
        2.0 = Center
        0.0 = Left
        """

        score_dict = {
            'left': 0,
            'left-center': 1,
            'least': 2,
            'right-center': 3,
            'right': 4
        }

        total_freq = 0
        total_score = 0
        for score_word, score_weight in score_dict.items():
            this_freq = grounding[word].cono[score_word]
            total_score += score_weight * this_freq
            total_freq += this_freq
    
    
        return total_score / total_freq
        
    if experiment_name == "dense":
        def calculate_cono(grounding, word):
            score_val = calculate_cono_raw(grounding, word)
            if score_val > 2.0:
                return 1
            else:
                return 0
    else:
        calculate_cono = calculate_cono_raw
            
else:
    if new_cr_corpus:
        def calculate_cono_raw(grounding, word):
            if grounding[word].majority_cono == 'D':
                return 1
            elif grounding[word].majority_cono == 'R':
                return 0
            else:
                assert False
        calculate_cono = calculate_cono_raw
    else:
        def calculate_cono_raw(grounding, word):
            return grounding[word]['R_ratio']
    
        def calculate_cono(grounding, word):
            skew = calculate_cono_raw(grounding, word)
            if skew < 0.5:
                return 0
            else:
                return 1
    
    def calculate_deno(grounding, word):
        if new_cr_corpus:
            topic = grounding[word].majority_deno
        else:
            topic = grounding[word]['majority_deno']
        
        if topic not in deno_topic_table:
            deno_topic_table[topic] = len(deno_topic_table)
            
        return deno_topic_table[topic]

pickfname = saved_pickle_name if use_saved_wordinfo else pickle_file
print("Loading Pickle file", pickfname)
with open(pickfname, 'rb') as f:
    pick_data = pickle.load(f)

w2id = pick_data["word_to_id"]
id2w = pick_data["id_to_word"]
ground = pick_data[grounding_name]
    
print("length of w2id:", len(w2id))
    
if not use_saved_wordinfo:
    new_pickle_root = {
        "word_to_id": w2id,
        "id_to_word": id2w
    }
    new_pickle_root[grounding_name] = ground
    print("Saving Pickle file", saved_pickle_name)
    with open(saved_pickle_name, "wb") as fout:
        pickle.dump(new_pickle_root, fout)


print("Loading text embedding", embedding_file)
raw_embed, out_of_vocabulary = Experiment.load_txt_embedding(embedding_file, pick_data["word_to_id"])

p_embedding = raw_embed.weight.numpy()

print("pretrained embedding shape", p_embedding.shape)


if experiment_name == "pos":
    master_pos_dict = get_full_pos_dict()
    global_pos_list_l = list(global_pos_list)
    pos_one_hot = [[] for i in global_pos_list_l]


query_conos = [] # 0 or 1 label
query_conos_raw = [] # sliding value
query_denos = []
original_words = []
filtered_embeddings = []
found_count = 0

deno_choices = []

#query_denos = {d:[] for d in deno_choices}

print("number of query words", len(w2id.keys()))


if True: # Filter embeddings
    for query_word in w2id.keys():
    #for query_word in albert_pick["word_to_id"].keys():

        if query_word in out_of_vocabulary:
            continue

        id = w2id[query_word]

        if experiment_name in ["cono", "sort", "dense"]:
            #if "_" not in query_word:
            #    # Only use compound words
            #    continue

            query_cono = calculate_cono(ground, query_word)
            query_conos.append(query_cono)
            
            query_cono_raw = calculate_cono_raw(ground, query_word)
            query_conos_raw.append(query_cono_raw)
            
            if not pn_corpus:
                query_deno = calculate_deno(ground, query_word)
                query_denos.append(query_deno)
            #query_deno = ground[query_word]['majority_deno']
        
            #for deno in deno_choices:
            #    if deno == query_deno:
            #        query_denos[deno].append(1)
            #    else:
            #        query_denos[deno].append(0)

        elif experiment_name == "pos":
            # POS experiment
            if "_" in query_word:
                # Skip compound words- won't be in dictionaries
                continue
            try:
                this_word_posset = master_pos_dict[query_word.lower()]
                if len(this_word_posset) != 1:
                    continue # Only use words with a single definition
                found_count += 1
            except KeyError:
                #this_word_posset = set()
                continue
    
            #alt_set = word_cat(query_word)
            #if alt_set:
            #    this_word_posset.add(alt_set)
            
            for idx, pos in enumerate(global_pos_list_l):
                if pos in this_word_posset:
                    pos_one_hot[idx].append(1)
                else:
                    pos_one_hot[idx].append(0)
    
        original_words.append(query_word)
        filtered_embeddings.append(p_embedding[id])

filtered_embeddings = np.array(filtered_embeddings)
unfiltered_embeddings = filtered_embeddings

print("filtered_embeddings shape", filtered_embeddings.shape)
print("Number of query denos", len(deno_topic_table))

if binarize_embeddings:
    filtered_embeddings = np.where(filtered_embeddings>0, 1, 0)
elif transform_embeddings:
    if randomize_embeddings:
        random.shuffle(query_conos)
        emb_shape = filtered_embeddings.shape
        filtered_embeddings = np.random.randn(*emb_shape)
    else:
        if use_pca:
            ca = PCA()
        else:
            ca = FastICA()
        ca.fit(filtered_embeddings)
    
        filtered_embeddings = ca.transform(filtered_embeddings)


def homogeneity_calc(embeddings, labels):
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    total_equal = 0
    total_vals = 0
    single_word_pct_list = []
    for word_offs, index_list in enumerate(indices):
        label_list = []
        for nbr_idx in index_list:
            if nbr_idx != word_offs:
                label_list.append(labels[nbr_idx])
        this_word_equal_cnt = len(list(filter(lambda x:x==labels[word_offs], label_list)))
        this_word_cnt = len(label_list)
        total_equal += this_word_equal_cnt
        total_vals += this_word_cnt
        this_word_pct = this_word_equal_cnt / this_word_cnt
        single_word_pct_list.append(this_word_pct)
    avg_homogeneity = total_equal / total_vals
    return avg_homogeneity, single_word_pct_list

def show_tsne(embeddings, labels, indices):
    """Show the TSNE projection of some embedding space, colored by labels"""
    tsne_proj_eng = TSNE(n_components=3, random_state=0)
    tsne_data = tsne_proj_eng.fit_transform(embeddings[indices,:])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_data[:,0], 
               tsne_data[:,1],
               tsne_data[:,2],
               c=np.array(labels)[indices])
    plt.axis('off')
    plt.show()

if experiment_name == "dense":
    
    use_ultradense = True # Use both Ultradense and classifier, otherwise class. only
    show_hom_tsne = False
    print_cono_wordorder = True
    print_cono_top10 = True # Limit to top/bottom 10, meaningful if print_cono_wordorder
    use_test_wordorder = False # Limit to test set, meaningful if print_cono_wordorder
    use_mse_loss = False # Otherwise, use UD loss
    use_sgd = True # Otherwise, Adam
    
    lr_choices = [0.05,0.005,0.0005]
    #lr_choices = [0.005]
    batch_size = 200
    num_epochs = 20
    offset_choice = 0
    ud_size = 1
    train_ratio = 0.9
    embedding_clipping = None # Set to e.g. 10000

    if embedding_clipping:
        filtered_embeddings = filtered_embeddings[:embedding_clipping]
        query_conos = query_conos[:embedding_clipping]
    
    if use_mse_loss:
        ud_size = max(ud_size, 2) # Need classifier in loop
    
    embedding_length = filtered_embeddings.shape[1]
    
    classifier_input_size = (embedding_length - ud_size) if use_ultradense else embedding_length
    
    num_labels = len(query_conos)
    num_1_labels = sum(query_conos)
    num_0_labels = num_labels - num_1_labels
    
    print("num_0_labels", num_0_labels, "num_1_labels", num_1_labels)
    
    base_label_cnt = min(num_1_labels, num_0_labels)
    
    query_conos_raw_np = np.array(query_conos_raw)
    query_conos_np = np.array(query_conos)
    if not pn_corpus:
        query_denos_np = np.array(query_denos)
    
    sort_lbl_idx = query_conos_np.argsort()
    zero_idces = sort_lbl_idx[:num_0_labels] # 0-indices
    one_idces = sort_lbl_idx[num_0_labels:] # 1-indicies
    
    np.random.shuffle(zero_idces)
    np.random.shuffle(one_idces)
    
    if False: # Normalize number of labels
        zero_idces = zero_idces[:base_label_cnt]
        one_idces = one_idces[:base_label_cnt]
    
    num_train_examples = int(train_ratio * base_label_cnt)
    num_test_examples = base_label_cnt - num_train_examples
    
    batches_per_epoch = int(num_train_examples/batch_size)
    
    holdout_set_zeroes = zero_idces[:num_test_examples]
    holdout_set_ones = one_idces[:num_test_examples]
    train_set_zeroes = zero_idces[num_test_examples:]
    train_set_ones = one_idces[num_test_examples:]
    
    train_indices = np.concatenate((train_set_zeroes,train_set_ones))
    test_indices = np.concatenate((holdout_set_zeroes,holdout_set_ones))
    
    train_embedding = filtered_embeddings[train_indices]
    test_embedding = filtered_embeddings[test_indices]
    train_query_conos_np = query_conos_np[train_indices]
    test_query_conos_np = query_conos_np[test_indices]

    if not pn_corpus:
        train_query_denos_np = query_denos_np[np.concatenate((train_set_zeroes,train_set_ones))] 
        test_query_denos_np = query_denos_np[np.concatenate((holdout_set_zeroes,holdout_set_ones))]
    
    # Data to be graphed
    axis_types = [
        "cono_loss", # Training loss of connotation classifier
        "cono_acc",  # Test accuracy of connotation classifier 

        ]

    if not pn_corpus:
        axis_types.extend([
            "deno_acc",   # Test accuracy of denotation classifier 
            "deno_loss", # Training loss of denotation classifier
        ])

    if use_ultradense:
        axis_types.extend([
            "ud_acc",        # Test accuracy of ultradense prediction
            "ud_correlation" # Test correlation of ultradense vectors with labels
        ])

    axes = {tname: [] for tname in axis_types}
    
    if use_ultradense and show_hom_tsne:
        print("Starting homogeneity")
        homogeneity, single_word_homogeneity_before = homogeneity_calc(filtered_embeddings, query_conos)
        print("Homogeneity was", homogeneity)
        if False: # Show tSNE as well
            print("Starting tSNE")
            indices = [i for i in range(0,len(filtered_embeddings),10)]
            show_tsne(filtered_embeddings, query_conos, indices)

    for learning_rate in lr_choices:
    
        print("Running model with lr=", learning_rate)
    
        axis_datapoints = {tname: [] for tname in axis_types}
        
        cono_model = Classifier(classifier_input_size, 2)
        cono_optimizer = torch.optim.Adam(cono_model.parameters(), lr=learning_rate)
        
        if not pn_corpus:
            deno_model = Classifier(classifier_input_size, len(deno_topic_table))
            deno_optimizer = torch.optim.Adam(deno_model.parameters(), lr=learning_rate)
            
        if use_ultradense:

            ud_model = UltraDense(embedding_length, offset_choice, ud_size, mse_loss=use_mse_loss)
            if use_sgd:
                ud_optimizer = torch.optim.SGD(ud_model.parameters(), lr=learning_rate)
                ud_scheduler = torch.optim.lr_scheduler.StepLR(ud_optimizer, step_size=1, gamma=0.95)
            else:
                ud_optimizer = torch.optim.Adam(ud_model.parameters(), lr=learning_rate)
            

            for e in range(num_epochs):
                # Copied from below
                zero_epoch_idx = np.random.choice(train_set_zeroes, batches_per_epoch * batch_size, False)
                one_epoch_idx = np.random.choice(train_set_ones, batches_per_epoch * batch_size, False)
                
                for b in range(batches_per_epoch):

                    # Also copied from below
                    batch_idx = np.concatenate((
                        zero_epoch_idx[b*batch_size:(b+1)*batch_size],
                        one_epoch_idx[b*batch_size:(b+1)*batch_size]
                        ))
                        
                    ud_optimizer.zero_grad()
                    
                    if use_mse_loss:
                        labels = query_conos_raw_np[batch_idx]
                    else:
                        labels = query_conos_np[batch_idx]
                    
                    model_result = ud_model(filtered_embeddings[batch_idx], labels)
                    ud_loss = ud_model.loss_func(model_result, labels)
                    ud_loss.backward()
                    ud_optimizer.step()
                    ud_model.orthogonalize()

                if use_sgd:
                    ud_scheduler.step()
            
                # Transform training space into ultradense, needed for threshold calculation
                ultra_dense_train_emb_space =  ud_model.apply_q(train_embedding)
                output_ultradense = ud_model.get_ultradense_1d_vector(ultra_dense_train_emb_space)

                # Determine threshold value for simple connotation prediction
                thresh, one_greater, one_values, zero_values = distro_thresh(output_ultradense, train_query_conos_np)

                ultra_dense_test_emb_space = ud_model.apply_q(test_embedding)
                output_ultradense_test = ud_model.get_ultradense_1d_vector(ultra_dense_test_emb_space)
                ud_predictions = preds_from_vector(thresh, one_greater, output_ultradense_test)
                ud_accuracy = sum(ud_predictions == test_query_conos_np) / len(test_query_conos_np)
                ud_corr, pval = stats.spearmanr(test_query_conos_np, output_ultradense_test)

                for data_item, data_value in (
                        ("ud_acc", ud_accuracy),
                        ("ud_correlation", ud_corr)):
                    axis_datapoints[data_item].append(data_value)
                print("Epoch: {} Ud acc {:.3f}, Ud Corr: {:.3f} Final Loss: {:.3f} Thresh: {:.3f}".format(
                    e, ud_accuracy, ud_corr, ud_loss, thresh))                

            ultra_dense_filtered_emb_space = ud_model.apply_q(filtered_embeddings)
            delete_slice = np.s_[offset_choice:offset_choice+ud_size]
            filtered_embeddings_2nd = np.delete(ultra_dense_filtered_emb_space, delete_slice, axis=-1)
            train_embedding_2nd = np.delete(ultra_dense_train_emb_space, delete_slice, axis=-1)
            test_embedding_2nd = np.delete(ultra_dense_test_emb_space, delete_slice, axis=-1)

            if show_hom_tsne:
                print("Starting homogeneity for ultradense")
                homogeneity, single_word_homogeneity_after = homogeneity_calc(filtered_embeddings_2nd, query_conos)
                print("Homogeneity for ultradense was", homogeneity)
                if False:
                    print("Starting tSNE")
                    indices = [i for i in range(0,len(filtered_embeddings),10)]
                    show_tsne(filtered_embeddings_2nd, query_conos, indices)
            

        else:
            # Definitions for 2nd stage (classifiers)
            train_embedding_2nd = train_embedding
            test_embedding_2nd = test_embedding
            filtered_embeddings_2nd = filtered_embeddings
                    
        print("Starting classifier filtered embedding shape", filtered_embeddings_2nd.shape) 
        for e in range(num_epochs):
        
            zero_epoch_idx = np.random.choice(train_set_zeroes, batches_per_epoch * batch_size, False)
            one_epoch_idx = np.random.choice(train_set_ones, batches_per_epoch * batch_size, False)
        
            total_conoloss = 0
            
            if not pn_corpus:
                total_denoloss = 0
            
            for b in range(batches_per_epoch):

                batch_idx = np.concatenate((
                    zero_epoch_idx[b*batch_size:(b+1)*batch_size],
                    one_epoch_idx[b*batch_size:(b+1)*batch_size]
                    ))
                cono_optimizer.zero_grad()
                if not pn_corpus:
                    deno_optimizer.zero_grad()


                classifier_input = filtered_embeddings_2nd[batch_idx]
                    
                cono_logits = cono_model(classifier_input, query_conos_np[batch_idx])
                cono_loss = cono_model.loss_func(cono_logits, query_conos_np[batch_idx])
                
                if not pn_corpus:
                    deno_logits = deno_model(classifier_input, query_denos_np[batch_idx])
                    deno_loss = deno_model.loss_func(deno_logits, query_denos_np[batch_idx])
                    
                
                total_conoloss += float(cono_loss)
                cono_loss.backward()
                cono_optimizer.step()
                
                if not pn_corpus:
                    total_denoloss += float(deno_loss)
                    deno_loss.backward()
                    deno_optimizer.step()
    
            cono_avg_loss_epoch = total_conoloss / batches_per_epoch
            cono_predictions = cono_model.test_forward(test_embedding_2nd)
            cono_accuracy = sum(cono_predictions == test_query_conos_np) / len(cono_predictions)
            if not pn_corpus:
                deno_avg_loss_epoch = total_denoloss / batches_per_epoch
                deno_predictions = deno_model.test_forward(test_embedding_2nd)
                deno_accuracy = sum(deno_predictions == test_query_denos_np) / len(deno_predictions)

            for data_item, data_value in (
                    ("cono_loss", cono_avg_loss_epoch),
                    ("cono_acc", cono_accuracy)):
                axis_datapoints[data_item].append(data_value)
                
            msg = "Epoch: {} Avg Loss (cono): {:.3f}, Accuracy (cono) {:.3f} ".format(e, cono_avg_loss_epoch, cono_accuracy)
                
            if not pn_corpus:
                for data_item, data_value in (
                        ("deno_loss", deno_avg_loss_epoch),
                        ("deno_acc", deno_accuracy)):
                    axis_datapoints[data_item].append(data_value)
                msg += "Avg Loss (deno): {:.3f}, , Accuracy (deno) {:.3f}".format(deno_avg_loss_epoch, deno_accuracy)
            print(msg)               

        if use_ultradense and print_cono_wordorder:
            wordorder = np.argsort(ud_model.get_ultradense_1d_vector(ultra_dense_filtered_emb_space))
            if use_test_wordorder:
                test_index_wordorder = np.array(list(filter(lambda y: y in test_indices, wordorder)))
                wordorder = test_index_wordorder
            
            top_ten_indices = wordorder[:10]
            bottom_ten_indices = wordorder[-10:]
            
            ud_1d_vector = ud_model.get_ultradense_1d_vector(ultra_dense_filtered_emb_space)
            
            def print_wordlist_line(word_idx):
                orig_word = original_words[word_idx]
                # Use trained transform to get coefficient
                coeff_value = ud_1d_vector[word_idx]
                label = calculate_cono(ground, orig_word)
                raw_label = calculate_cono_raw(ground, orig_word)
                base_string = "{:20.20s} val:{:+5.3f} lab:{}".format(orig_word, coeff_value, label)
                
                if pn_corpus:
                    leaning_string = " raw:{:+5.3f} l:{:4d} lc:{:4d} m:{:4d} rc:{:4d} r:{:4d}".format(
                        raw_label,
                        ground[orig_word].cono.get('left', 0),
                        ground[orig_word].cono.get('left-center', 0),
                        ground[orig_word].cono.get('least', 0),
                        ground[orig_word].cono.get('right-center', 0),
                        ground[orig_word].cono.get('right', 0)
                    )
                    base_string += leaning_string
                
                if show_hom_tsne:
                    homogeneity_pct_before =  single_word_homogeneity_before[word_idx] * 100
                    homogeneity_pct_after = single_word_homogeneity_after[word_idx] * 100
                    pct_string = " hbef:{:3.1f} haft:{:3.1f}".format(homogeneity_pct_before, homogeneity_pct_after)
                    base_string += pct_string
                    
                
                print(base_string)
                
            if print_cono_top10: # Top ten
                print("\nTop ten ultradense words:\n")
                for word_idx in top_ten_indices:
                    print_wordlist_line(word_idx)
                print("\nBottom ten ultradense words:\n")
                for word_idx in bottom_ten_indices:
                    print_wordlist_line(word_idx)
            else: # Entire list
                for word_idx in wordorder:
                    print_wordlist_line(word_idx)
                

        for data_item, data_list in axis_datapoints.items():
            axes[data_item].append(data_list)
    
#     fig = plt.figure()
#     ax = plt.axes()
#     n_bins = 100
#     ax.hist(zero_values, bins=n_bins)
#     ax.hist(one_values, bins=n_bins)
#     ax.set_xlabel('Connotation component values')
#     ax.set_ylabel('Count')
#     plt.title("Distribution of Rep./Demo. connotation in ultradense component")
#     plt.legend()
#     plt.show()
    
    for data_item, axis_list in axes.items():
    
        ylabel_str, title_str = {
            "cono_loss": ("Loss", "Connotation Loss"),
            "deno_loss": ("Loss", "Denotation Loss"),
            "cono_acc": ("Accuracy", "Connotation Accuracy"),
            "deno_acc": ("Accuracy", "Denotation Accuracy"),
            "ud_acc": ("Accuracy", "Ultradense Connotation Accuracy"),
            "ud_correlation": ("Spearman Correlation", "Ultradense Connotation Correlation")
            }[data_item]
        fig = plt.figure()
        ax = plt.axes()
        for lr, lax in zip(lr_choices, axis_list):
            ax.plot(lax, label=str(lr))
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel_str)
        if data_item in ["cono_loss", "deno_loss"]:
            plt.ylim((-2, 5))
        plt.title(title_str)
        plt.legend()
        plt.show()

elif experiment_name == "cono":
    rvals = []
    for emb_pos in range(filtered_embeddings.shape[1]):
    
        if binarize_embeddings:
            rval, pval = stats.pointbiserialr(filtered_embeddings[:,emb_pos], query_conos)
        else:
            rval, pval = stats.spearmanr(query_conos, filtered_embeddings[:,emb_pos])
        
        rvals.append(tuple(np.nan_to_num((rval, emb_pos, pval))))

    rvals.sort()
    print("min, max cono corr:", rvals[0], rvals[-1])

#     for deno in deno_choices:
#         rvals = []
#         for emb_pos in range(filtered_embeddings.shape[1]):
#             total_nonzero =  len(np.nonzero(filtered_embeddings[:,emb_pos])[0])
#             if use_avg_prec:
#                 avg_prec_pos = average_precision_score(query_denos[deno], filtered_embeddings[:,emb_pos])
#                 rvals.append(tuple(np.nan_to_num((avg_prec_pos, emb_pos, total_nonzero))))
#             else:
#                 rval, pval = stats.pointbiserialr(query_denos[deno], filtered_embeddings[:,emb_pos])
#                 rvals.append(tuple(np.nan_to_num((rval, emb_pos, total_nonzero))))
#         rvals.sort()
#         if use_avg_prec:
#             print("avg prec. deno corr ({:45s})".format(deno), rvals[-1])
#         else:
#             print("min, max deno corr ({:45s})".format(deno), rvals[0], rvals[-1])

    #plt.scatter(query_conos, filtered_embeddings[:,284], s=4)
    #plt.xlabel("Connotation Ratio")
    #plt.ylabel("Component 284 from PCA")
    #plt.show()
elif experiment_name == "pos":

    for idx, pos in enumerate(global_pos_list_l):
        true_ratio = sum(pos_one_hot[idx]) / len(pos_one_hot[idx])
        print("{} True: {:.1f}%".format(pos, true_ratio * 100))

    num_pca_components_displayed = 10
    correlation_matrix = [[] for i in range(num_pca_components_displayed)]
    for row_idx, matrix_row in enumerate(correlation_matrix):
        # Each row is PCA vec
        for col_idx, pos in enumerate(global_pos_list_l):
            # Columns are POS
            if use_avg_prec:
                avg_prec = average_precision_score(pos_one_hot[col_idx], filtered_embeddings[:,row_idx])
                matrix_row.append(avg_prec)
            else:
                rval, pval = stats.pointbiserialr(pos_one_hot[col_idx], filtered_embeddings[:,row_idx])
                matrix_row.append(rval)
    correlation_matrix = np.array(correlation_matrix)
    np.nan_to_num(correlation_matrix, copy=False)
    print(correlation_matrix)
    
    print("min", correlation_matrix.min(), "max", correlation_matrix.max())
    
    prefix = "PCA" if transform_embeddings else "W2V"
    pcalabels = [prefix + str(i) for i in range(num_pca_components_displayed)]
    poslabels = global_pos_list_l
    
    
    fig, ax = plt.subplots()

    im, cbar = heatmap(correlation_matrix, pcalabels, poslabels, ax=ax,
                       cmap="RdYlBu", cbarlabel="Correlation", vmin=-1.0, vmax=1.0)
    #texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    plt.show()
elif experiment_name == "pairs":
    #pair_argsort = None
    fig_words = []
    fig_rand = []
    for w1, w2 in cherry_pairs:
    

        word_count = len(w2id)
        randid1, randid2 = random.sample(range(word_count), 2)
        try:
            wordid1, wordid2 = w2id[w1], w2id[w2]
        except KeyError:
            print("Skipping", w1, w2, "since not found")
            continue
    

        for id1, id2, isRand, tgt_list in [(randid1, randid2, True, fig_rand), (wordid1, wordid2, False, fig_words)]:
            diff_vector = p_embedding[id1] - p_embedding[id2]
            #if pair_argsort is None:
            #    pair_argsort = np.argsort(diff_vector)
            diff_vector.sort()
            diff_vector = np.flip(diff_vector)
            diff_vector = diff_vector[:300]
            tgt_list.append(diff_vector)

    fig = plt.figure()
    ax = plt.axes()
    #ax.plot(diff_vector[pair_argsort])
    for diff_vector in fig_rand:
        ax.plot(diff_vector)
    ax.set_xlabel('Component')
    ax.set_ylabel('Difference')
    plt.title("Random pair differences")
    plt.ylim(-0.15, 0.15)
    plt.show()
    
    fig = plt.figure()
    ax = plt.axes()
    #ax.plot(diff_vector[pair_argsort])
    for diff_vector in fig_words:
        ax.plot(diff_vector)
    ax.set_xlabel('Component')
    ax.set_ylabel('Difference')
    plt.title("Luntz-esque differences")
    plt.ylim(-0.15, 0.15)
    plt.show()
    
# Old way: throwing up individual charts

#         for id1, id2, isRand in [(randid1, randid2, True), (wordid1, wordid2, False)]:
#             fig = plt.figure()
#             diff_vector = filtered_embeddings[id1] - filtered_embeddings[id2]
#             #if pair_argsort is None:
#             #    pair_argsort = np.argsort(diff_vector)
#             ax = plt.axes()
#             #ax.plot(diff_vector[pair_argsort])
#             ax.plot(diff_vector)
#             ax.set_xlabel('Component')
#             ax.set_ylabel('Difference')
#             if isRand:
#                 plt.title("Random components")
#             else:
#                 plt.title("{} vs {}".format(w1, w2))
#             plt.show()
        

else: # "sort"
    
    num_sort_components = 8 # Start at the left (might not be principal)
    
    sorted_idx = np.argsort(-1.0 * filtered_embeddings,axis=0)
    
    max_word_len = len(max(w2id.keys(), key=len))
    
    print("max word len", max_word_len)
    

    target_idx = 1 # PCA component offset
    display_len = 3000
    display_step = 20
    
    idces = sorted_idx[0:display_len:display_step,target_idx]
    #idces = sorted_idx[:,target_idx]
    
    if True: # Experiment 1 - sorted list display
    
        if True: # Use variance to pick components
            variance_list = np.var(filtered_embeddings, axis=0)
            component_idces = np.argsort(variance_list)[-num_sort_components:]
        else:
            component_idces = range(num_sort_components)
        print("Indices", component_idces)
    
        for r in sorted_idx:
            for c in r[component_idces]:
                sys.stdout.write("{:20.20s} ".format(id2w[c]))
            print("")        
    else:
        fig = plt.figure()
        if False: # Experiment 2 i.e. one-hot vector
            target_deno = 'Immigration'
            #component0vals = filtered_embeddings[idces,target_idx]
            component0vals = np.array(query_denos[target_deno])[idces]
            ax = plt.axes()
            N = 50 # Window size
            ax.scatter(x=range(len(component0vals)), y=np.convolve(component0vals, np.ones((N,))/N, mode='same'))
            ax.set_xlabel('Word ID')
            ax.set_ylabel('Moving average of one-hot denotation vector')
        else: #3D plot of TSNE, experiment 3
    
            tsne_proj_eng = TSNE(n_components=3, random_state=0)
            tsne_data = tsne_proj_eng.fit_transform(unfiltered_embeddings)
    
            cmhot = plt.get_cmap("RdYlGn")
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(tsne_data[idces,0], 
                       tsne_data[idces,1],
                       tsne_data[idces,2],
                       c=filtered_embeddings[idces,target_idx], 
                       cmap=cmhot)
        plt.show()
