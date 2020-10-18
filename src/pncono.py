import pickle

from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from utils.experiment import Experiment
from ultradense import UltraDense
 
# Experiment control knobs
experiment_name = "dense" # One of: cono, pos, sort, pairs, dense
use_avg_prec = False
binarize_embeddings = False # Implies !transform_embeddings
transform_embeddings = False
use_pca = False # Otherwise, ICA. Only applies if transform_embeddings is True.
pn_corpus = False # Partisan News if True otherwise, Congressional Record
use_saved_wordinfo = True


saved_pickle_name = "intermediate.pickle"
if pn_corpus:
    pickle_file = "../data/ready/PN_proxy/train.pickle"
    embedding_file = "../data/pretrained_word2vec/PN_proxy_method_B.txt"
    embedding_file = "../data/pretrained_word2vec/PN_heading_proxy.txt"

else:
    pickle_file = "../data/ready/CR_proxy/train.pickle"
    embedding_file = "../data/pretrained_word2vec/CR_proxy.txt"    


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


for query_word in w2id.keys():

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
    
    batch_size = 120
    
    filtered_embeddings = filtered_embeddings[:40000]
    query_conos = query_conos[:40000]
    
    embedding_length = filtered_embeddings.shape[1]
    
    num_labels = len(query_conos)
    num_1_labels = sum(query_conos)
    num_0_labels = num_labels - num_1_labels
    
    base_label_cnt = min(num_1_labels, num_0_labels)
    batches_per_epoch = int(base_label_cnt/batch_size)
    
    query_conos_np = np.array(query_conos)
    
    sort_lbl_idx = query_conos_np.argsort()
    zero_idces = sort_lbl_idx[:num_0_labels] # 0-indices
    one_idces = sort_lbl_idx[num_0_labels:] # 1-indicies
        

    
    offset_choice = 3
    model = UltraDense(embedding_length, offset_choice)
    #optimizer = torch.optim.Adam(model.parameters(), lr=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    print("num_labels", num_labels, "num_1_labels", num_1_labels)
    
    
    
    num_epochs = 30
    for e in range(num_epochs):
        print("starting epoch", e)
        
        zero_epoch_idx = np.random.choice(zero_idces, batches_per_epoch * batch_size, False)
        one_epoch_idx = np.random.choice(one_idces, batches_per_epoch * batch_size, False)
        
        total_intloss = 0
        for b in range(batches_per_epoch):

            batch_idx = np.concatenate((
                zero_epoch_idx[b*batch_size:(b+1)*batch_size],
                one_epoch_idx[b*batch_size:(b+1)*batch_size]
                ))
            optimizer.zero_grad()    
            Lcts_result, Lct_result = model(filtered_embeddings[batch_idx], query_conos_np[batch_idx])
            loss = model.loss_func(Lcts_result, Lct_result)
            floatloss = float(loss)
            if b % 10 == 0:
                print("batch", b)
                print("loss", floatloss)
            total_intloss += floatloss
            loss.backward()
            optimizer.step()
    
            #print("Q before ortho:", model.Q)
    
            model.orthogonalize()
    
    
        scheduler.step()
        
        #print("Q after ortho:", model.Q)
    
        output_ultradense = model.apply_q(filtered_embeddings)[:,offset_choice]
        
        #print("Epoch", e, output_ultradense)
        #sorted_result_idx = np.argsort(output_ultradense)

        #print("Sorted component", output_ultradense[sorted_result_idx])
        #print("Sorted labels", labels[sorted_result_idx])

        rval, pval = stats.spearmanr(query_conos, output_ultradense)
        print("Correlation:", rval)
        print("avg loss:", total_intloss / batches_per_epoch)

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
    for w1, w2 in cherry_pairs:
    
        word_count = len(w2id)
        randid1, randid2 = random.sample(range(word_count), 2)
        wordid1, wordid2 = w2id[w1], w2id[w2]
    
        for id1, id2, isRand in [(randid1, randid2, True), (wordid1, wordid2, False)]:
            fig = plt.figure()
            diff_vector = filtered_embeddings[id1] - filtered_embeddings[id2]
            #if pair_argsort is None:
            #    pair_argsort = np.argsort(diff_vector)
            ax = plt.axes()
            #ax.plot(diff_vector[pair_argsort])
            ax.plot(diff_vector)
            ax.set_xlabel('Component')
            ax.set_ylabel('Difference')
            if isRand:
                plt.title("Random components")
            else:
                plt.title("{} vs {}".format(w1, w2))
            plt.show()
        

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
