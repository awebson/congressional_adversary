import argparse
import random
import pickle
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Set, Tuple, List, Dict, Counter, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import homogeneity_score
from tqdm import tqdm
import editdistance  # for excluding trivial nearest neighbors

from utils.experiment import Experiment
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor
from evaluations.word_similarity import all_wordsim as word_sim

random.seed(42)
torch.manual_seed(42)


class Decomposer(nn.Module):

    def __init__(
            self,
            config: 'DecomposerConfig',
            data: 'LabeledDocuments'):
        super().__init__()
        self.negative_sampling_probs = data.negative_sampling_probs
        self.num_negative_samples = config.num_negative_samples
        self.max_adversary_loss = config.max_adversary_loss
        self.init_embedding(config, data.word_to_id)
        # sent_repr_size = 300

        # Dennotation Loss: Skip-Gram Negative Sampling
        # self.deno_decoder = nn.Linear(sent_repr_size, num_deno_classes)
        # assert self.deno_decoder[0].in_features == sent_repr_size
        # assert self.deno_decoder[-2].out_features == num_deno_classes

        # Connotation Loss: Party Classifier
        # self.cono_decoder = nn.Linear(sent_repr_size, num_cono_classes)
        num_cono_classes = 2
        self.cono_decoder = config.cono_architecture
        assert self.cono_decoder[-2].out_features == num_cono_classes

        self.delta = config.delta
        self.gamma = config.gamma
        self.rho = config.rho
        self.max_adversary_loss = config.max_adversary_loss
        self.device = config.device
        self.to(self.device)

        # Initailize neighbor cono homogeneity eval partisan vocabulary
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.grounding: Dict[str, Counter[str]] = data.grounding
        self.init_grounding()

    def init_embedding(
            self,
            config: 'DecomposerConfig',
            word_to_id: Dict[str, int]
            ) -> None:
        if config.pretrained_embedding is not None:
            self.embedding = Experiment.load_txt_embedding(
                config.pretrained_embedding, word_to_id)
        else:
            self.embedding = nn.Embedding(len(word_to_id), config.embed_size)
            init_range = 1.0 / config.embed_size
            nn.init.uniform_(self.embedding.weight.data, -init_range, init_range)
        self.embedding.weight.requires_grad = not config.freeze_embedding

        # freeze a copy of the pretrained embedding
        self.pretrained_embed = nn.Embedding.from_pretrained(self.embedding.weight)
        self.pretrained_embed.weight.requires_grad = False

    def init_grounding(self) -> None:
        dev_path = Path('../data/ellie/partisan_sample_val.cr.txt')  # TODO renew sampling
        with open(dev_path) as file:
            self.dev_ids = torch.tensor(
                [self.word_to_id[word.strip()] for word in file],
                device=self.device)

        def pretrained_neighbors(
                query_ids: Vector,
                top_k: int = 10
                ) -> Dict[int, Set[int]]:
            deno_grounding: Dict[int, Set[int]] = {}
            # self.pretrained_embed = self.pretrained_embed.to(self.device)
            # with torch.no_grad():
            for qid in query_ids:
                qv = self.pretrained_embed(qid)
                qid = qid.item()
                qw = self.id_to_word[qid]
                cos_sim = F.cosine_similarity(qv.unsqueeze(0), self.pretrained_embed.weight)
                cos_sim, neighbor_ids = cos_sim.topk(k=top_k + 5, dim=-1)
                neighbor_ids = [
                    nid for nid in neighbor_ids.tolist()
                    if editdistance.eval(qw, self.id_to_word[nid]) > 3]
                deno_grounding[qid] = set(neighbor_ids[:top_k])
            return deno_grounding

        self.deno_grounding: Dict[int, Set[int]] = pretrained_neighbors(self.dev_ids)

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
        cono_loss = nn.functional.cross_entropy(cono_logits, cono_labels)

        overcorrect_loss = 1 - F.cosine_similarity(
            word_vecs, self.pretrained_embed(seq_word_ids), dim=-1).mean()

        if self.max_adversary_loss:
            if self.gamma < 0:  # remove connotation
                cono_loss = torch.clamp(cono_loss, max=self.max_adversary_loss)
            else:  # remove denotation
                deno_loss = torch.clamp(deno_loss, max=self.max_adversary_loss)

        decomposer_loss = (
            self.delta * deno_loss
            + self.gamma * cono_loss
            + self.rho * overcorrect_loss)

        if recompose:
            return decomposer_loss, deno_loss, cono_loss, word_vecs
        else:
            return decomposer_loss, deno_loss, cono_loss, overcorrect_loss

    def skip_gram_loss(
            self,
            center_word_ids: Vector,
            true_context_ids: Vector,
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

        # if error_analysis_path:
        #     analysis_file = open(error_analysis_path, 'w')
        #     analysis_file.write('pred_conf\tpred\tlabel_conf\tlabel\tseq\n')
        #     output = []
        #     for pred_confs, pred_id, label_id, seq_ids in zip(
        #             deno_conf, deno_predictions, deno_labels, seq_word_ids):
        #         pred_conf = f'{pred_confs[pred_id].item():.4f}'
        #         label_conf = f'{pred_confs[label_id].item():.4f}'
        #         pred = self.id_to_deno[pred_id.item()]
        #         label = self.id_to_deno[label_id.item()]
        #         seq = ' '.join([self.id_to_word[i.item()] for i in seq_ids])
        #         output.append((pred_conf, pred, label_conf, label, seq))
        #     # output.sort(key=lambda t: t[1], reverse=True)
        #     for stuff in output:
        #         analysis_file.write('\t'.join(stuff) + '\n')

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
        cos_sim, neighbor_ids = cos_sim.topk(k=top_k, dim=-1)
        if verbose:
            return cos_sim[:, 1:], neighbor_ids[:, 1:]
        else:  # excludes the first neighbor, which is always the query itself
            return neighbor_ids[:, 1:]

    @staticmethod
    def discretize_cono(skew: float) -> int:
        if skew < 0.5:
            return 0
        else:
            return 1

    def id_to_cono(self, word_id: int) -> int:
        return self.discretize_cono(
            self.grounding[self.id_to_word[word_id]]['R_ratio'])

    # @staticmethod
    # def discretize_cono(skew: float) -> int:
    #     if skew < 0.2:
    #         return 0
    #     elif skew < 0.8:
    #         return 1
    #     else:
    #         return 2

    def homemade_homogeneity(
            self,
            query_ids: Vector,
            top_k: int = 10
            ) -> Tuple[float, float]:
        # extra top_k buffer for excluding edit distance neighbors
        top_neighbor_ids = self.nearest_neighbors(query_ids, top_k + 5)
        deno_homogeneity = []
        # cono_homogeneity = []
        cono_homogeneity_discrete = []
        for query_index, neighbor_ids in enumerate(top_neighbor_ids):
            query_id = query_ids[query_index].item()
            query_word = self.id_to_word[query_id]
            neighbor_ids = [
                nid for nid in neighbor_ids.tolist()
                if editdistance.eval(query_word, self.id_to_word[nid]) > 3]
            neighbor_ids = neighbor_ids[:top_k]

            if len(neighbor_ids) == 0:
                # print(query_word, [self.id_to_word[i.item()] for i in top_neighbor_ids[query_index]])
                continue

            query_deno: Set[int] = self.deno_grounding[query_id]
            overlap = len([nid for nid in neighbor_ids if nid in query_deno])
            deno_homogeneity.append(overlap / len(neighbor_ids))

            query_cono = self.id_to_cono(query_id)
            same_cono = len(
                [nid for nid in neighbor_ids
                 if self.id_to_cono(nid) == query_cono])
            cono_homogeneity_discrete.append(same_cono / len(neighbor_ids))

        return np.mean(deno_homogeneity), np.mean(cono_homogeneity_discrete)

    def SciPy_homogeneity(
            self,
            query_ids: Vector,
            top_k: int = 10
            ) -> Tuple[float, float]:
        top_neighbor_ids = self.nearest_neighbors(query_ids, top_k)
        cluster_ids = []
        true_cono_labels = []
        deno_homogeneity = []
        for query_index, neighbor_ids in enumerate(top_neighbor_ids):
            query_id = query_ids[query_index].item()
            query_word = self.id_to_word[query_id]
            neighbor_ids = [
                nid for nid in neighbor_ids.tolist()
                if editdistance.eval(query_word, self.id_to_word[nid]) > 3]
            neighbor_ids = neighbor_ids[:top_k]

            if len(neighbor_ids) == 0:
                continue

            query_deno: Set[int] = self.deno_grounding[query_id]
            overlap = len([nid for nid in neighbor_ids if nid in query_deno])
            deno_homogeneity.append(overlap / len(neighbor_ids))

            cluster_ids += [query_index] * len(neighbor_ids)
            true_cono_labels += [self.discrete_cono[nid] for nid in neighbor_ids]

        deno_homogeneity = np.mean(deno_homogeneity)
        cono_homogeneity = homogeneity_score(true_cono_labels, cluster_ids)
        return deno_homogeneity, cono_homogeneity

    def old_SciPy_homogeneity(
            self,
            query_ids: Vector,
            eval_deno: bool,
            top_k: int = 5
            ) -> Tuple[float, float]:
        top_neighbor_ids = self.nearest_neighbors(query_ids, top_k)
        cluster_ids = []
        true_labels = []
        # naive_homogeneity = []
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
            # naive_homogeneity.append(num_same_label / top_k)
        homogeneity = homogeneity_score(true_labels, cluster_ids)
        return homogeneity  # completness?

    def old_homemade_homogeneity(
            self,
            query_ids: Vector,
            eval_deno: bool,
            top_k: int = 5
            ) -> Tuple[float, float]:
        top_neighbor_ids = self.nearest_neighbors(query_ids, top_k)
        # cluster_ids = []
        # true_labels = []
        naive_homogeneity = []
        for query_index, sorted_target_indices in enumerate(top_neighbor_ids):
            query_id = query_ids[query_index].item()
            query_word = self.id_to_word[query_id]
            # cluster_id = query_index
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
                # cluster_ids.append(cluster_id)
                # true_labels.append(neighbor_label)
                if neighbor_label == query_label:
                    num_same_label += 1
            # End Looping Nearest Neighbors
            naive_homogeneity.append(num_same_label / top_k)
        return np.mean(naive_homogeneity)

    def tabulate(
            self,
            query_ids: Vector,
            suffix: str,
            rounding: int = 4
            ) -> Dict[str, float]:
        row = {}
        Hdeno, Hcono = self.homemade_homogeneity(query_ids, top_k=10)
        row['Hdeno'] = Hdeno
        row['Hcono'] = Hcono
        row['Intra Hd - Hc'] = Hdeno - Hcono
        if not suffix:
            return {key: round(val, rounding) for key, val in row.items()}
        else:
            return {key + suffix: round(val, rounding) for key, val in row.items()}



class LabeledDocuments(Dataset):

    def __init__(self, config: 'DecomposerConfig'):
        corpus_path = os.path.join(config.input_dir, 'train_data.pickle')
        with open(corpus_path, 'rb') as corpus_file:
            preprocessed = pickle.load(corpus_file)
        self.word_to_id = preprocessed['word_to_id']
        self.id_to_word = preprocessed['id_to_word']
        self.documents: List[List[int]] = preprocessed['documents']
        self.cono_labels: List[int] = preprocessed['cono_labels']
        self.negative_sampling_probs = torch.tensor(preprocessed['negative_sampling_probs'])
        self.grounding: Dict[str, Counter[str]] = preprocessed['grounding']
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

    def __len__(self) -> int:
        return len(self.documents)

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
        self.data = LabeledDocuments(config)
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

    def train(self) -> None:
        model = self.model
        config = self.config
        grad_clip = config.clip_grad_norm
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
                seq_word_ids = batch[2].to(self.device)
                cono_labels = batch[3].to(self.device)

                model.zero_grad()
                L_decomp, l_deno, l_cono, l_overcorrect = model(
                    center_word_ids, context_word_ids, seq_word_ids, cono_labels)
                L_decomp.backward()

                nn.utils.clip_grad_norm_(self.model.cono_decoder.parameters(), grad_clip)
                self.cono_optimizer.step()

                nn.utils.clip_grad_norm_(self.model.embedding.parameters(), grad_clip)
                self.decomposer_optimizer.step()

                if batch_index % config.update_tensorboard == 0:
                    cono_accuracy = self.model.accuracy(seq_word_ids, cono_labels)
                    stats = {
                        'Decomposer/deno_loss': l_deno,
                        'Decomposer/cono_loss': l_cono,
                        'Decomposer/overcorrect_loss': l_overcorrect,
                        # 'Decomposer/accuracy_train_deno': deno_accuracy,
                        'Decomposer/accuracy_train_cono': cono_accuracy,
                        'Decomposer/combined_loss': L_decomp
                    }
                    self.update_tensorboard(stats)
                # if config.print_stats and batch_index % config.print_stats == 0:
                #     self.print_stats(epoch_index, batch_index, stats)
                if batch_index % config.eval_dev_set == 0:
                    self.validation()
                self.tb_global_step += 1
            # End Batches
            self.print_timestamp(epoch_index)
            self.auto_save(epoch_index)
        # End Epochs

    def validation(self) -> None:
        model = self.model
        Hdeno, Hcono = model.homemade_homogeneity(model.dev_ids)

        mean_delta, abs_rhos = word_sim.mean_delta(
            model.embedding.weight, model.pretrained_embed.weight,
            model.id_to_word, reduce=False)
        cos_sim = F.cosine_similarity(
            model.embedding.weight, model.pretrained_embed.weight).mean()
        self.update_tensorboard({
            'Decomposed Space/Neighbor Overlap': Hdeno,
            'Decomposed Space/Party Homogeneity': Hcono,
            # 'Decomposed Space/Party Homogeneity SciPy': Hcono_SP,
            'Decomposed Space/Overlap - Party': Hdeno - Hcono,

            'Decomposed Space/rho difference cf pretrained': mean_delta,
            'Decomposed Space/MTurk-771': abs_rhos[0],
            'Decomposed Space/cosine cf pretrained': cos_sim
        })



@dataclass
class DecomposerConfig():
    # Essential
    input_dir: Path = Path('../data/processed/CR_skip')
    output_dir: Path = Path('../results/debug')
    device: torch.device = torch.device('cuda')
    debug_subset_corpus: Optional[int] = None
    # dev_holdout: int = 5_000
    # test_holdout: int = 10_000
    num_dataloader_threads: int = 6  # NOTE
    pin_memory: bool = True

    beta: float = 10
    decomposed_size: int = 300
    delta: float = 1  # denotation classifier weight 𝛿
    gamma: float = 1  # connotation classifier weight 𝛾
    rho: float = 100
    max_adversary_loss: Optional[float] = 10

    architecture: str = 'L1'
    dropout_p: float = 0
    batch_size: int = 2
    embed_size: int = 300
    num_epochs: int = 10
    # encoder_update_cycle: int = 1  # per batch
    # decoder_update_cycle: int = 1  # per batch

    # pretrained_embedding: Optional[str] = None
    pretrained_embedding: Optional[Path] = Path('../data/pretrained_word2vec/for_real.txt')
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
    auto_save_per_epoch: Optional[int] = 1
    auto_save_if_interrupted: bool = False

    def __post_init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-i', '--input-dir', action='store', type=Path)
        parser.add_argument(
            '-o', '--output-dir', action='store', type=Path)
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
            '-pe', '--pretrained-embedding', action='store', type=Path)
        parser.parse_args(namespace=self)

        if self.architecture == 'L1':
            # self.deno_architecture = nn.Sequential(
            #     nn.Linear(300, 41),
            #     nn.SELU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 2),
                nn.SELU())
        elif self.architecture == 'L2':
            # self.deno_architecture = nn.Sequential(
            #     nn.Linear(300, 300),
            #     nn.SELU(),
            #     nn.Linear(300, 41),
            #     nn.SELU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 2),
                nn.SELU())
        elif self.architecture == 'L4':
            # self.deno_architecture = nn.Sequential(
            #     nn.Linear(300, 300),
            #     nn.SELU(),
            #     nn.AlphaDropout(p=self.dropout_p),
            #     nn.Linear(300, 300),
            #     nn.SELU(),
            #     nn.AlphaDropout(p=self.dropout_p),
            #     nn.Linear(300, 300),
            #     nn.SELU(),
            #     nn.Linear(300, 41),
            #     nn.SELU())
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
