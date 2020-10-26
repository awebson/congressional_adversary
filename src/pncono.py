import pickle

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
from ultradense import UltraDense
from classifier import Classifier
 
# Experiment control knobs
experiment_name = "dense" # One of: cono, pos, sort, pairs, dense
use_avg_prec = False
binarize_embeddings = False # Implies !transform_embeddings
transform_embeddings = False
use_pca = False # Otherwise, ICA. Only applies if transform_embeddings is True.
pn_corpus = False # Partisan News if True otherwise, Congressional Record
use_saved_wordinfo = True


albert_pick_file = "../albert_wordlist.pickle"

with open(albert_pick_file, 'rb') as albert_file:
    albert_pick = pickle.load(albert_file)

saved_pickle_name = "intermediate.pickle"
if pn_corpus:
    pickle_file = "../data/ready/PN_proxy/train.pickle"
    embedding_file = "../data/pretrained_word2vec/PN_proxy_method_B.txt"
    #embedding_file = "../data/pretrained_word2vec/PN_heading_proxy.txt"

else:

    #embedding_file = "/Users/tberckma/Documents/Brown/Research/AlbertProject/newcong_ad_data/data/pretrained_word2vec/for_real_SGNS_method_B.txt"
    #pickle_file = "/Users/tberckma/Documents/Brown/Research/AlbertProject/congressional_adversary/data/ready/CR_proxy/train.pickle"

    pickle_file = "../data/ready/CR_proxy/train.pickle"
    embedding_file = "../data/pretrained_word2vec/CR_proxy.txt"    


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
    def calculate_cono(grounding, word):
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
else:
    def calculate_cono(grounding, word):
        if grounding[word].majority_cono == 'D':
            return 1
        elif grounding[word].majority_cono == 'R':
            return 0
        else:
            assert False

print("Loading Pickle file")
with open(saved_pickle_name if use_saved_wordinfo else pickle_file, 'rb') as f:
    pick_data = pickle.load(f)

w2id = pick_data["word_to_id"]
id2w = pick_data["id_to_word"]
ground = pick_data['ground']
    
print("length of w2id:", len(w2id))
    
if not use_saved_wordinfo:
    new_pickle_root = {
        "ground": ground,
        "word_to_id": w2id,
        "id_to_word": id2w
    }
    print("Saving Pickle file")
    with open(saved_pickle_name, "wb") as fout:
        pickle.dump(new_pickle_root, fout)


print("Loading text embedding")
raw_embed, out_of_vocabulary = Experiment.load_txt_embedding(embedding_file, pick_data["word_to_id"])

p_embedding = raw_embed.weight.numpy()

print("pretrained embedding shape", p_embedding.shape)


if experiment_name == "pos":
    master_pos_dict = get_full_pos_dict()
    global_pos_list_l = list(global_pos_list)
    pos_one_hot = [[] for i in global_pos_list_l]


query_conos = []
filtered_embeddings = []
found_count = 0

deno_choices = []

query_denos = {d:[] for d in deno_choices}

print("number of query words", len(w2id.keys()))


if True: # Filter embeddings
    #for query_word in w2id.keys():
    for query_word in albert_pick["word_to_id"].keys():

        if query_word in out_of_vocabulary:
            continue

        id = w2id[query_word]

        if experiment_name in ["cono", "sort", "dense"]:
            #if "_" not in query_word:
            #    # Only use compound words
            #    continue

            query_cono = calculate_cono(ground, query_word)
            query_conos.append(query_cono)
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
    
        filtered_embeddings.append(p_embedding[id])

filtered_embeddings = np.array(filtered_embeddings)
unfiltered_embeddings = filtered_embeddings

print("filtered_embeddings shape", filtered_embeddings.shape)

if binarize_embeddings:
    filtered_embeddings = np.where(filtered_embeddings>0, 1, 0)
elif transform_embeddings:
    if use_pca:
        ca = PCA()
    else:
        ca = FastICA()
    ca.fit(filtered_embeddings)
    
    filtered_embeddings = ca.transform(filtered_embeddings)


if experiment_name == "dense":
    
    classifier_type = False # As opposed to ultradense
    
    lr_choices = [0.5,0.05,0.005,0.0005]
    batch_size = 200
    num_epochs = 20
    offset_choice = 3
    train_ratio = 0.9
    embedding_clipping = None # Set to e.g. 10000
    
    if embedding_clipping:
        filtered_embeddings = filtered_embeddings[:embedding_clipping]
        query_conos = query_conos[:embedding_clipping]
    
    embedding_length = filtered_embeddings.shape[1]
    
    num_labels = len(query_conos)
    num_1_labels = sum(query_conos)
    num_0_labels = num_labels - num_1_labels
    
    base_label_cnt = min(num_1_labels, num_0_labels)
    
    query_conos_np = np.array(query_conos)
    
    sort_lbl_idx = query_conos_np.argsort()
    zero_idces = sort_lbl_idx[:num_0_labels] # 0-indices
    one_idces = sort_lbl_idx[num_0_labels:] # 1-indicies
    
    np.random.shuffle(zero_idces)
    np.random.shuffle(one_idces)
    
    num_train_examples = int(train_ratio * base_label_cnt)
    num_test_examples = base_label_cnt - num_train_examples
    
    batches_per_epoch = int(num_train_examples/batch_size)
    
    holdout_set_zeroes = zero_idces[:num_test_examples]
    holdout_set_ones = one_idces[:num_test_examples]
    train_set_zeroes = zero_idces[num_test_examples:]
    train_set_ones = one_idces[num_test_examples:]
    
    train_embedding = filtered_embeddings[np.concatenate((train_set_zeroes,train_set_ones))]
    test_embedding = filtered_embeddings[np.concatenate((holdout_set_zeroes,holdout_set_ones))]
    train_query_conos_np = query_conos_np[np.concatenate((train_set_zeroes,train_set_ones))]
    test_query_conos_np = query_conos_np[np.concatenate((holdout_set_zeroes,holdout_set_ones))]
    
    print("Num 0 labels:", num_0_labels)
    print("Num 1 labels:", num_1_labels)
        
    loss_axes = []
    corr_axes = []
    acc_axes = []
    testcorr_axes = []    
    
    for learning_rate in lr_choices:
    
        print("Running model with lr=", learning_rate)
    
        loss_per_epoch = []
        corr_per_epoch = []
        testcorr_per_epoch = []
        acc_par_batch = []
        
        if classifier_type:
            model = Classifier(embedding_length, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            model = UltraDense(embedding_length, offset_choice)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        for e in range(num_epochs):
        
            zero_epoch_idx = np.random.choice(train_set_zeroes, batches_per_epoch * batch_size, False)
            one_epoch_idx = np.random.choice(train_set_ones, batches_per_epoch * batch_size, False)
        
            total_intloss = 0
            for b in range(batches_per_epoch):

                batch_idx = np.concatenate((
                    zero_epoch_idx[b*batch_size:(b+1)*batch_size],
                    one_epoch_idx[b*batch_size:(b+1)*batch_size]
                    ))
                optimizer.zero_grad()
                if classifier_type:
                    logits = model(filtered_embeddings[batch_idx], query_conos_np[batch_idx])
                    loss = model.loss_func(logits, query_conos_np[batch_idx])
                else: 
                    Lcts_result, Lct_result = model(filtered_embeddings[batch_idx], query_conos_np[batch_idx])
                    loss = model.loss_func(Lcts_result, Lct_result)
                floatloss = float(loss)
                total_intloss += floatloss
                loss.backward()
                optimizer.step()
                if not classifier_type:
                    model.orthogonalize()
    
            if not classifier_type:
                scheduler.step()
                output_ultradense = model.apply_q(train_embedding)[:,offset_choice]
                rval, pval = stats.spearmanr(train_query_conos_np, output_ultradense)
                corr_per_epoch.append(rval)

            avg_loss_epoch = total_intloss / batches_per_epoch
            loss_per_epoch.append(avg_loss_epoch)
            
            if classifier_type:
                predictions = model.test_forward(test_embedding)
                accuracy = sum(predictions == test_query_conos_np) / len(predictions)
                print("Epoch: {} Accuracy: {:.3f} (test): {:.3f} , Avg Loss: {:.3f} ".format(e, accuracy, 0.0, avg_loss_epoch))
                acc_par_batch.append(accuracy)
            else:
                output_ultradense_test = model.apply_q(test_embedding)[:,offset_choice]
                test_rval, test_pval = stats.spearmanr(test_query_conos_np, output_ultradense_test)
                print("Epoch: {} Correlation: {:.3f} (test): {:.3f} , Avg Loss: {:.3f} ".format(e, rval, test_rval, avg_loss_epoch))
                testcorr_per_epoch.append(test_rval)

        loss_axes.append(loss_per_epoch)
        corr_axes.append(corr_per_epoch)
        acc_axes.append(acc_par_batch)
        testcorr_axes.append(testcorr_per_epoch)
        
    
    fig = plt.figure()
    ax = plt.axes()
    for lr, lax in zip(lr_choices, loss_axes):
        ax.plot(lax, label=str(lr))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.title("Loss by learning rate")
    plt.legend()
    plt.show()
    
    
    fig = plt.figure()
    ax = plt.axes()
    for lr, lax in zip(lr_choices, corr_axes):
        ax.plot(lax, label=str(lr))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Spearman Correlation')
    plt.title("Correlation by learning rate")
    plt.legend()
    plt.show()
    
    fig = plt.figure()
    ax = plt.axes()
    for lr, lax in zip(lr_choices, testcorr_axes):
        ax.plot(lax, label=str(lr))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Spearman Correlation')
    plt.title("Correlation (test) by learning rate")
    plt.legend()
    plt.show()
    
    fig = plt.figure()
    ax = plt.axes()
    for lr, lax in zip(lr_choices, acc_axes):
        ax.plot(lax, label=str(lr))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    plt.title("Accuracy by learning rate")
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
