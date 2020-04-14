import argparse
import random
import pickle
import os
from dataclasses import dataclass
from typing import Tuple, List, Dict, Counter, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.utils import rnn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import homogeneity_score
from tqdm import tqdm
import editdistance  # for excluding trivial nearest neighbors

# from sklearn.metrics import pairwise
# from sklearn.metrics import confusion_matrix
# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.set()

from utils.experiment import Experiment
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor

random.seed(42)
torch.manual_seed(42)


class Decomposer(nn.Module):

    def __init__(
            self,
            config: 'DecomposerConfig',
            data: 'LabeledSentences'):
        super().__init__()

        vocab_size = len(data.word_to_id)
        if config.pretrained_embedding is not None:
            config.embed_size = data.pretrained_embedding.shape[1]
            self.embedding = nn.Embedding.from_pretrained(data.pretrained_embedding)
        else:
            self.embedding = nn.Embedding(vocab_size, config.embed_size)
            init_range = 1.0 / config.embed_size
            nn.init.uniform_(self.embedding.weight.data, -init_range, init_range)
        self.embedding.weight.requires_grad = not config.freeze_embedding

        self.negative_sampling_probs = data.negative_sampling_probs
        self.num_negative_samples = config.num_negative_samples
        num_cono_classes = 2
        # sent_repr_size = 300

        # Dennotation Loss: Skip-Gram Negative Sampling
        # self.deno_decoder = nn.Linear(sent_repr_size, num_deno_classes)
        # assert self.deno_decoder[0].in_features == sent_repr_size
        # assert self.deno_decoder[-2].out_features == num_deno_classes

        # Connotation Loss: Party Classifier
        # self.cono_decoder = nn.Linear(sent_repr_size, num_cono_classes)
        self.cono_decoder = config.cono_architecture
        assert self.cono_decoder[-2].out_features == num_cono_classes

        self.delta = config.delta
        self.gamma = config.gamma
        self.beta = config.beta
        self.device = config.device
        self.to(self.device)

        # Initailize neighbor cono homogeneity eval partisan vocabulary
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        # self.grounding = data.grounding

    def forward(
            self,
            center_word_ids: Vector,
            true_context_ids: Vector,
            # negative_context_ids: Matrix,
            seq_word_ids: Matrix,
            cono_labels: Vector,
            recompose: bool = False,
            ) -> Scalar:
        word_vecs: R3Tensor = self.embedding(seq_word_ids)
        seq_repr: Matrix = torch.mean(word_vecs, dim=1)

        deno_loss = self.skip_gram_loss(
            center_word_ids, true_context_ids)  # , negative_context_ids)

        cono_logits = self.cono_decoder(seq_repr)

        # import IPython
        # IPython.embed()
        cono_loss = nn.functional.cross_entropy(cono_logits, cono_labels)

        decomposer_loss = (self.delta * deno_loss +
                           self.gamma * cono_loss +
                           self.beta)
        if recompose:
            return decomposer_loss, deno_loss, cono_loss, word_vecs
        else:
            return decomposer_loss, deno_loss, cono_loss

    def skip_gram_loss(
            self,
            center_word_ids: Vector,
            true_context_ids: Vector,
            # negative_context_ids: Matrix
            ) -> Scalar:
        """Faster but less readable."""
        negative_context_ids = torch.multinomial(
            self.negative_sampling_probs,
            len(true_context_ids) * self.num_negative_samples,
            replacement=True
        ).view(len(true_context_ids), self.num_negative_samples).to(self.device)

        center = self.embedding(center_word_ids)
        true_context = self.embedding(true_context_ids)
        negative_context = self.embedding(negative_context_ids)
        # center = self.deno_decoder(encoded_center)
        # true_context = self.deno_decoder(encoded_true_context)
        # negative_context = self.deno_decoder(encoded_negative_context)

        # batch_size * embed_size
        objective = torch.sum(
            torch.mul(center, true_context),  # Hadamard product
            dim=1)  # be -> b
        objective = nn.functional.logsigmoid(objective)

        # batch_size * num_negative_samples * embed_size
        # negative_context: bne
        # center: be -> be1
        negative_objective = torch.bmm(  # bne, be1 -> bn1
            negative_context, center.unsqueeze(2)
            ).squeeze()  # bn1 -> bn
        negative_objective = nn.functional.logsigmoid(-negative_objective)
        negative_objective = torch.sum(negative_objective, dim=1)  # bn -> b
        return -torch.mean(objective + negative_objective)

    def predict(self, seq_word_ids: Vector) -> Vector:
        self.eval()
        with torch.no_grad():
            word_vecs: R3Tensor = self.embedding(seq_word_ids)
            seq_repr: Matrix = torch.mean(word_vecs, dim=1)
            cono = self.cono_decoder(seq_repr)
            cono_conf = nn.functional.softmax(cono, dim=1)
        self.train()
        return cono_conf

    def accuracy(
            self,
            seq_word_ids: Matrix,
            cono_labels: Vector,
            error_analysis_path: Optional[str] = None
            ) -> Tuple[float, float]:
        cono_conf = self.predict(seq_word_ids)
        cono_predictions = cono_conf.argmax(dim=1)
        # # Random Guess Baseline
        # deno_predictions = torch.randint_like(deno_labels, high=len(self.deno_to_id))
        # # Majority Class Baseline
        # majority_label = self.deno_to_id['Health']
        # deno_predictions = torch.full_like(deno_labels, majority_label)

        cono_correct_indicies = cono_predictions.eq(cono_labels)
        cono_accuracy = cono_correct_indicies.float().mean().item()

        if error_analysis_path:
            analysis_file = open(error_analysis_path, 'w')
            analysis_file.write('pred_conf\tpred\tlabel_conf\tlabel\tseq\n')
            output = []
            for pred_confs, pred_id, label_id, seq_ids in zip(
                    deno_conf, deno_predictions, deno_labels, seq_word_ids):
                pred_conf = f'{pred_confs[pred_id].item():.4f}'
                label_conf = f'{pred_confs[label_id].item():.4f}'
                pred = self.id_to_deno[pred_id.item()]
                label = self.id_to_deno[label_id.item()]
                seq = ' '.join([self.id_to_word[i.item()] for i in seq_ids])
                output.append((pred_conf, pred, label_conf, label, seq))
            # output.sort(key=lambda t: t[1], reverse=True)
            for stuff in output:
                analysis_file.write('\t'.join(stuff) + '\n')

        # if error_analysis_path:  # confusion  matrix
        #     cf_mtx = confusion_matrix(deno_labels.cpu(), deno_predictions.cpu())
        #     fig, ax = plt.subplots(figsize=(20, 20))
        #     sns.heatmap(
        #         cf_mtx, annot=True, robust=True, ax=ax, cbar=False, fmt='d', linewidths=.5,
        #         mask=np.equal(cf_mtx, 0),
        #         xticklabels=self.graph_labels, yticklabels=self.graph_labels)
        #     ax.set_xlabel('Predicted Label')
        #     ax.set_ylabel('True Label')
        #     with open(error_analysis_path, 'wb') as file:
        #         fig.savefig(file, dpi=300, bbox_inches='tight')
        return cono_accuracy

    def nearest_neighbors(
            self,
            query_ids: Vector,
            top_k: int = 10,
            verbose: bool = False,
            ) -> List[Vector]:
        with torch.no_grad():
            query_vectors = self.embedding(query_ids)
        try:
            cos_sim = nn.functional.cosine_similarity(
                query_vectors.unsqueeze(1),
                self.embedding.weight.unsqueeze(0),
                dim=2)
        except RuntimeError:  # insufficient GPU memory
            cos_sim = torch.stack([
                nn.functional.cosine_similarity(
                    q.unsqueeze(0), self.embedding.weight)
                for q in query_vectors])
        cos_sim, neighbor_ids = cos_sim.topk(k=top_k + 10, dim=-1)
        if verbose:
            return cos_sim, neighbor_ids
        else:
            return neighbor_ids

    # @staticmethod
    # def discretize_cono(skew: float) -> int:
    #     if skew < 0.5:
    #         return 0
    #     else:
    #         return 1

    @staticmethod
    def discretize_cono(skew: float) -> int:
        if skew < 0.2:
            return 0
        elif skew < 0.8:
            return 1
        else:
            return 2

    def NN_cluster_homogeneity(
            self,
            query_ids: Vector,
            eval_deno: bool,
            top_k: int = 5
            ) -> Tuple[float, float]:
        top_neighbor_ids = self.nearest_neighbors(query_ids, top_k)
        cluster_ids = []
        true_labels = []
        naive_homogeneity = []
        for query_index, sorted_target_indices in enumerate(top_neighbor_ids):
            query_id = query_ids[query_index].item()
            query_word = self.id_to_word[query_id]
            cluster_id = query_index

            num_same_label = 0
            if eval_deno:
                query_label = self.deno_to_id[self.grounding[query_word]['majority_deno']]
            else:
                query_label = self.discretize_cono(self.grounding[query_word]['R_ratio'])

            num_neighbors = 0
            for sort_rank, target_id in enumerate(sorted_target_indices):
                target_id = target_id.item()
                if num_neighbors == top_k:
                    break
                if query_id == target_id:
                    continue
                target_word = self.id_to_word[target_id]
                if editdistance.eval(query_word, target_word) < 3:
                    continue
                num_neighbors += 1

                if eval_deno:
                    neighbor_label = self.deno_to_id[
                        self.grounding[target_word]['majority_deno']]
                else:
                    neighbor_label = self.discretize_cono(
                        self.grounding[target_word]['R_ratio'])
                cluster_ids.append(cluster_id)
                true_labels.append(neighbor_label)

                if neighbor_label == query_label:
                    num_same_label += 1
            # End Looping Nearest Neighbors
            naive_homogeneity.append(num_same_label / top_k)

        homogeneity = homogeneity_score(true_labels, cluster_ids)
        return homogeneity  # completness?, np.mean(naive_homogeneity)

    def homemade_heterogeneity(
            self,
            query_ids: Vector,
            top_neighbor_ids: List[Vector],
            top_k: int = 10
            ) -> float:
        homogeneity = []
        for query_index, sorted_target_indices in enumerate(top_neighbor_ids):
            query_id = query_ids[query_index].item()
            query_word = self.id_to_word[query_id]
            num_neighbors = 0

            query_R_ratio = self.grounding[query_word]['R_ratio']
            freq_ratio_distances = []
            for sort_rank, target_id in enumerate(sorted_target_indices):
                target_id = target_id.item()
                if num_neighbors == top_k:
                    break
                if query_id == target_id:
                    continue
                # target_id = target_ids[target_index]  # target is always all embed
                target_words = self.id_to_word[target_id]
                if editdistance.eval(query_word, target_words) < 3:
                    continue
                num_neighbors += 1

                # Homemade continuous heterogeneity
                target_R_ratio = self.grounding[target_words]['R_ratio']
                # freq_ratio_distances.append((target_R_ratio - query_R_ratio) ** 2)
                freq_ratio_distances.append(abs(target_R_ratio - query_R_ratio))
            # homogeneity.append(np.sqrt(np.mean(freq_ratio_distances)))
            homogeneity.append(np.mean(freq_ratio_distances))
        return np.mean(homogeneity)


class LabeledSentences(Dataset):

    def __init__(self, config: 'DecomposerConfig'):
        corpus_path = os.path.join(config.input_dir, 'train_data.pickle')
        with open(corpus_path, 'rb') as corpus_file:
            preprocessed = pickle.load(corpus_file)
        self.word_to_id = preprocessed['word_to_id']
        self.id_to_word = preprocessed['id_to_word']
        self.documents: List[List[int]] = preprocessed['documents']
        self.cono_labels: List[int] = preprocessed['cono_labels']
        self.negative_sampling_probs = torch.tensor(preprocessed['negative_sampling_probs'])
        # self.grounding: Dict[str, Counter[str]] = preprocessed['grounding']
        self.window_radius = config.window_radius
        self.num_negative_samples = config.num_negative_samples

        # HACK
        self.fixed_sent_len = 15
        self.min_sent_len = 5

        # deno/cono -> Counter[word]
        # self.counts: Dict[str, Counter[str]] = preprocessed['counts']

        # self.train_seq: List[List[int]] = preprocessed['train_sent_word_ids']
        # self.train_cono_labels: List[int] = preprocessed['train_cono_labels']

        # self.dev_seq = rnn.pad_sequence(
        #     [torch.tensor(seq) for seq in preprocessed['dev_sent_word_ids']],
        #     batch_first=True)
        # self.dev_cono_labels = torch.tensor(preprocessed['dev_cono_labels'])

        del preprocessed

        if config.pretrained_embedding is not None:
            if config.pretrained_embedding.endswith('txt'):
                self.pretrained_embedding: Matrix = Experiment.load_embedding(
                    config.pretrained_embedding, self.word_to_id)
            elif config.pretrained_embedding.endswith('pt'):
                self.pretrained_embedding = torch.load(
                    config.pretrained_embedding,
                    map_location=config.device)['model'].embedding.weight
            else:
                raise ValueError('Unknown pretrained embedding format.')

    def __len__(self) -> int:
        return len(self.documents)

    # def init_negative_sampling(
    #             self,
    #             word_frequency: CounterType[str],
    #             word_to_id: Dict[str, int]
    #             ) -> None:
    #         """
    #         A smoothed unigram distribution.
    #         A simplified case of Noise Contrastive Estimate.
    #         where the seemingly arbitrary number of 0.75 is from Mikolov 2014.
    #         """
    #         # message = (
    #         #     'As of PyTorch 1.1, torch.distributions.categorical.Categorical '
    #         #     'is very slow, so a third-party alternative is necessary for now: '
    #         #     'https://pypi.org/project/pytorch-categorical/'
    #         # )
    #         # try:
    #         #     import pytorch_categorical
    #         # except ModuleNotFoundError as error:
    #         #     print(message)
    #         #     raise error
    #         cumulative_freq = sum(freq ** 0.75 for freq in word_frequency.values())
    #         # debug missing vocab
    #         # for word, freq in word_frequency.items():
    #         #     if word not in word_to_id:
    #         #         print(freq, word)

    #         categorical_dist_probs: Dict[int, float] = {
    #             word_to_id[word]: (freq ** 0.75) / cumulative_freq
    #             for word, freq in word_frequency.items()
    #         }
    #         vocab_size = len(word_to_id)
    #         categorical_dist_probs: Vector = torch.tensor([
    #             # categorical_dist_probs[word_id]  # strict
    #             categorical_dist_probs.get(word_id, 0)  # prob = 0 if missing vocab
    #             for word_id in range(vocab_size)
    #         ])
    #         self.categorical_dist_probs = categorical_dist_probs
    #         # self.negative_sampling_dist = torch.distributions.categorical.Categorical(
    #         #     categorical_dist_probs)

    #         # self.negative_sampling_dist = torch.distributions.multinomial.Multinomial(
    #         #      categorical_dist_probs)
    #         # self.negative_sampling_dist = pytorch_categorical.Categorical(
    #         #     categorical_dist_probs, self.device)

    # def faux_sent_tokenize(self, word_ids: List[int]) -> Iterable[List[int]]:
    #     start_index = 0
    #     while (start_index + self.fixed_sent_len) < (len(word_ids) - 1):
    #         yield word_ids[start_index:start_index + self.fixed_sent_len]
    #         start_index += self.fixed_sent_len

    #     trailing_words = word_ids[start_index:-1]
    #     if len(trailing_words) >= self.min_sent_len:
    #         yield trailing_words

    def __getitem__(
            self,
            index: int
            ) -> Tuple[
            List[int],
            List[int],
            # List[List[int]],
            List[List[int]],
            int]:
        """process one document into skip-gram pairs and faux sentences"""
        doc: List[int] = self.documents[index]

        # For Denotation
        # parsing (center, context) word id pairs
        center_word_ids: List[int] = []
        context_word_ids: List[int] = []
        for center_index, center_word_id in enumerate(doc):
            left_index = max(center_index - self.window_radius, 0)
            right_index = min(center_index + self.window_radius, len(doc) - 1)
            context_word_id: List[int] = (
                doc[left_index:center_index] +
                doc[center_index + 1:right_index + 1])
            context_word_ids += context_word_id
            center_word_ids += [center_word_id] * len(context_word_id)
            # labels += [self.id_to_label[center_word_id]] * len(context_word_id)

        # negative_context_ids = torch.multinomial(
        #     self.negative_sampling_probs,
        #     len(context_word_ids) * self.num_negative_samples,
        #     replacement=True
        # ).view(len(context_word_ids), self.num_negative_samples)

        # For Connotation, split a document into faux sentences
        faux_sentences: List[List[int]] = []
        start_index = 0
        while (start_index + self.fixed_sent_len) < (len(doc) - 1):
            faux_sentences.append(
                doc[start_index:start_index + self.fixed_sent_len])
            start_index += self.fixed_sent_len

        trailing_words = doc[start_index:-1]
        if len(trailing_words) >= self.min_sent_len:
            faux_sentences.append(trailing_words)

        return (
            center_word_ids,
            context_word_ids,
            # negative_context_ids,
            faux_sentences,
            self.cono_labels[index])

    @staticmethod
    def collate(
            processed_docs: List[Tuple[
            List[int],
            List[int],
            # List[List[int]],
            List[List[int]],
            int]]
            ) -> Tuple[Vector, Vector, Matrix, Vector]:
        """flatten indeterminate docs full of data into batches"""
        # seq_word_ids = [torch.tensor(w) for w, _, _ in processed_docs]
        # deno_labels = torch.tensor([d for _, d, _ in processed_docs])
        # cono_labels = torch.tensor([c for _, _, c in processed_docs])
        center_ids = []
        context_ids = []
        # negative_ids = []
        seq_word_ids: List[Vector] = []
        cono_labels = []
        for center, context, seqs, cono_label in processed_docs:
            center_ids += center
            context_ids += context
            # negative_ids += neg_sample
            seq_word_ids += [torch.tensor(seq) for seq in seqs]
            cono_labels += [cono_label] * len(seqs)

        # import IPython
        # IPython.embed()
        return (
            torch.tensor(center_ids),
            torch.tensor(context_ids),
            # torch.tensor(negative_ids),
            rnn.pad_sequence(seq_word_ids, batch_first=True),
            torch.tensor(cono_labels))


class DecomposerExperiment(Experiment):

    def __init__(self, config: 'DecomposerConfig'):
        super().__init__(config)
        self.data = LabeledSentences(config)
        self.dataloader = DataLoader(
            self.data,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.data.collate,
            num_workers=config.num_dataloader_threads,
            pin_memory=config.pin_memory)
        self.model = Decomposer(config, self.data)

        # For Debugging
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)  # param.data)

        # self.deno_optimizer = config.optimizer(
        #     self.model.deno_decoder.parameters(),
        #     lr=config.learning_rate)
        self.cono_optimizer = config.optimizer(
            self.model.cono_decoder.parameters(),
            lr=config.learning_rate)
        self.decomposer_optimizer = config.optimizer(
            self.model.embedding.parameters(),
            lr=config.learning_rate)

        self.to_be_saved = {
            'config': self.config,
            'model': self.model}
        # self.custom_stats_format = (
        #     'ℒ = {Decomposer/combined_loss:.3f}\t'
        #     'd = {Decomposer/deno_loss:.3f}\t'
        #     'c = {Decomposer/cono_loss:.3f}\t'
        #     'decomp = {Recomposer/recomp:.3f}\t'
        #     'cono accuracy = {Evaluation/connotation_accuracy:.2%}'
        # )

    def _train(
            self,
            center_word_ids: Vector,
            context_word_ids: Vector,
            # negative_ids: Matrix,
            seq_word_ids: Matrix,
            cono_labels: Vector,
            update_encoder: bool = True,
            update_decoder: bool = True
            ) -> Tuple[float, float, float]:
        grad_clip = self.config.clip_grad_norm
        self.model.zero_grad()
        L_decomp, l_deno, l_cono = self.model(
            center_word_ids, context_word_ids, # negative_ids,
            seq_word_ids, cono_labels)

        if update_decoder:
            # l_deno.backward(retain_graph=True)
            # nn.utils.clip_grad_norm_(self.model.deno_decoder.parameters(), grad_clip)
            # self.deno_optimizer.step()

            self.model.zero_grad()
            l_cono.backward(retain_graph=update_encoder)
            nn.utils.clip_grad_norm_(self.model.cono_decoder.parameters(), grad_clip)
            self.cono_optimizer.step()

        if update_encoder:
            self.model.zero_grad()
            L_decomp.backward()
            nn.utils.clip_grad_norm_(self.model.embedding.parameters(), grad_clip)
            self.decomposer_optimizer.step()
        return L_decomp.item(), l_deno.item(), l_cono.item()

    def train(self) -> None:
        config = self.config
        # # For debugging
        # self.save_everything(
        #     os.path.join(self.config.output_dir, f'init.pt'))
        # raise SystemExit

        if not config.print_stats:
            epoch_pbar = tqdm(range(1, config.num_epochs + 1), desc=config.output_dir)
        else:
            epoch_pbar = tqdm(range(1, config.num_epochs + 1), desc='Epochs')
        for epoch_index in epoch_pbar:
            if not config.print_stats:
                batches = enumerate(self.dataloader)
            else:
                batches = tqdm(
                    enumerate(self.dataloader),
                    total=len(self.dataloader),
                    mininterval=config.progress_bar_refresh_rate,
                    desc='Batches')
            for batch_index, batch in batches:
                center_word_ids = batch[0].to(self.device)
                context_word_ids = batch[1].to(self.device)
                # negative_ids = batch[2].to(self.device)
                seq_word_ids = batch[2].to(self.device)
                cono_labels = batch[3].to(self.device)

                L_decomp, l_deno, l_cono = self._train(
                    center_word_ids, context_word_ids,  # negative_ids,
                    seq_word_ids, cono_labels)

                if batch_index % config.update_tensorboard == 0:
                    cono_accuracy = self.model.accuracy(seq_word_ids, cono_labels)
                    stats = {
                        'Decomposer/deno_loss': l_deno,
                        'Decomposer/cono_loss': l_cono,
                        # 'Decomposer/accuracy_train_deno': deno_accuracy,
                        'Decomposer/accuracy_train_cono': cono_accuracy,
                        'Decomposer/combined_loss': L_decomp
                    }
                    self.update_tensorboard(stats)
                # if config.print_stats and batch_index % config.print_stats == 0:
                #     self.print_stats(epoch_index, batch_index, stats)
                # if batch_index % config.eval_dev_set == 0:
                #     # deno_accuracy, cono_accuracy = self.model.accuracy(
                #     #     self.data.dev_seq.to(self.device),
                #     #     self.data.dev_deno_labels.to(self.device),
                #     #     self.data.dev_cono_labels.to(self.device))

                #     # D_h = self.model.homemade_homogeneity_discrete(self.model.Dem_ids)
                #     # R_hd = self.model.homemade_homogeneity_discrete(self.model.GOP_ids)
                #     # N_hd = self.model.homemade_homogeneity_discrete(self.model.neutral_ids)

                #     # D_h = self.model.homemade_heterogeneity(self.model.Dem_ids)
                #     # R_hc = self.model.homemade_heterogeneity(self.model.GOP_ids)
                #     # N_hc = self.model.homemade_heterogeneity(self.model.neutral_ids)
                #     self.update_tensorboard({
                #         # 'Denotation Decomposer/nonpolitical_word_sim_cf_pretrained': deno_check,
                #         # 'Denotation Decomposer/accuracy_dev_deno': deno_accuracy,
                #         # 'Connotation Decomposer/nonpolitical_word_sim_cf_pretrained': cono_check,
                #         'Denotation Decomposer/accuracy_dev_cono': cono_accuracy,
                #         # 'Intrinsic Evaluation/Dem_neighbor_homogenity': D_h,
                #         # 'Intrinsic Evaluation/GOP_neighbor_homogenity': R_hd,
                #         # 'Intrinsic Evaluation/GOP_neighbor_heterogeneity_continous': R_hc,
                #         # 'Intrinsic Evaluation/neutral_neighbor_homogenity': N_hd,
                #         # 'Intrinsic Evaluation/neutral_neighbor_heterogeneity_continous': N_hc,
                #         # 'Recomposer/nonpolitical_word_sim_cf_pretrained': recomp_check}
                #     })
                self.tb_global_step += 1
            # End Batches
            # self.lr_scheduler.step()
            self.print_timestamp(epoch_index)
            self.auto_save(epoch_index)

            # if config.export_error_analysis:
            #     if (epoch_index % config.export_error_analysis == 0
            #             or epoch_index == 1):
            #         # self.model.all_vocab_connotation(os.path.join(
            #         #     config.output_dir, f'vocab_cono_epoch{epoch_index}.txt'))
            #         analysis_path = os.path.join(
            #             config.output_dir, f'error_analysis_epoch{epoch_index}.tsv')
            #         deno_accuracy, cono_accuracy = self.model.accuracy(
            #             self.data.dev_seq.to(self.device),
            #             self.data.dev_deno_labels.to(self.device),
            #             self.data.dev_cono_labels.to(self.device),
            #             error_analysis_path=analysis_path)
        # End Epochs


@dataclass
class DecomposerConfig():
    # Essential
    input_dir: str = '../data/processed/labeled_documents/for_real'
    output_dir: str = '../results/debug'
    device: torch.device = torch.device('cuda')
    debug_subset_corpus: Optional[int] = None
    # dev_holdout: int = 5_000
    # test_holdout: int = 10_000
    num_dataloader_threads: int = 12  # NOTE
    pin_memory: bool = True

    beta: float = 10
    decomposed_size: int = 300
    delta: float = 1  # denotation classifier weight 𝛿
    gamma: float = -1  # connotation classifier weight 𝛾

    architecture: str = 'L1'
    dropout_p: float = 0
    batch_size: int = 2
    embed_size: int = 300
    num_epochs: int = 10
    # encoder_update_cycle: int = 1  # per batch
    # decoder_update_cycle: int = 1  # per batch

    # pretrained_embedding: Optional[str] = None
    pretrained_embedding: Optional[str] = '../data/pretrained_word2vec/for_real.txt'
    freeze_embedding: bool = False  # NOTE
    window_radius: int = 5
    num_negative_samples: int = 10
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    # optimizer: torch.optim.Optimizer = torch.optim.SGD
    learning_rate: float = 1e-4
    # momentum: float = 0.5
    # lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    # num_prediction_classes: int = 5
    clip_grad_norm: float = 10.0

    # Housekeeping
    export_error_analysis: Optional[int] = 1  # per epoch
    update_tensorboard: int = 1000  # per batch
    print_stats: Optional[int] = 10_000  # per batch
    eval_dev_set: int = 10_000  # per batch
    progress_bar_refresh_rate: int = 5  # per second
    suppress_stdout: bool = False  # during hyperparameter tuning
    reload_path: Optional[str] = None
    clear_tensorboard_log_in_output_dir: bool = True
    delete_all_exisiting_files_in_output_dir: bool = False
    auto_save_per_epoch: Optional[int] = 10
    auto_save_if_interrupted: bool = False

    def __post_init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-i', '--input-dir', action='store', type=str)
        parser.add_argument(
            '-o', '--output-dir', action='store', type=str)
        parser.add_argument(
            '-gpu', '--device', action='store', type=str)

        parser.add_argument(
            '-a', '--architecture', action='store', type=str)
        parser.add_argument(
            '-d', '--delta', action='store', type=float)
        parser.add_argument(
            '-g', '--gamma', action='store', type=float)

        parser.add_argument(
            '-lr', '--learning-rate', action='store', type=float)
        parser.add_argument(
            '-bs', '--batch-size', action='store', type=int)
        parser.add_argument(
            '-ep', '--num-epochs', action='store', type=int)
        parser.add_argument(
            '-pe', '--pretrained-embedding', action='store', type=str)
        parser.parse_args(namespace=self)

        if self.architecture == 'L1':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 41),
                nn.SELU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 2),
                nn.SELU())
        elif self.architecture == 'L2':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 41),
                nn.SELU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 2),
                nn.SELU())
        elif self.architecture == 'L4':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.AlphaDropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.SELU(),
                nn.AlphaDropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 41),
                nn.SELU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.AlphaDropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.SELU(),
                nn.AlphaDropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 2),
                nn.SELU())
        else:
            raise ValueError('Unknown architecture argument.')


def main() -> None:
    config = DecomposerConfig()
    black_box = DecomposerExperiment(config)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    main()
