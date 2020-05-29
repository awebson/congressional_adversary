import argparse
import pickle
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Set, Tuple, List, Dict, Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import numpy as np  # for intrinsic eval
import editdistance  # for excluding trivial nearest neighbors
from sklearn.metrics import homogeneity_score

from data import Sentence, LabeledDoc, GroundedWord
from evaluations.word_similarity import all_wordsim as word_sim
from utils.experiment import Experiment
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor

random.seed(42)
torch.manual_seed(42)

class Decomposer(nn.Module):

    def __init__(
            self,
            config: 'DecomposerConfig',
            data: 'LabeledDocuments'):
        super().__init__()
        self.num_negative_samples = config.num_negative_samples
        self.negative_sampling_probs = data.negative_sampling_probs
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.delta = config.delta
        self.gamma = config.gamma
        self.init_embedding(config, data.word_to_id)

        # Dennotation Loss: Skip-Gram Negative Sampling
        # self.cosine_loss = nn.CosineEmbeddingLoss()

        # Connotation Loss: Party Classifier
        self.cono_decoder = config.cono_decoder
        self.num_cono_classes = config.num_cono_classes

        self.device = config.device
        self.to(self.device)
        self.ground: Dict[str, GroundedWord] = data.ground
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
        # Connotation grounding
        id_to_cono = []
        id_to_freq = []
        # is_unigram = []
        # Itereate in order of word ids
        for wid in range(self.embedding.num_embeddings):
            word = self.ground[self.id_to_word[wid]]
            id_to_cono.append(word.cono_PMI)
            id_to_freq.append(word.cono_freq)
            # if '_' in word.word:
            #     is_unigram.append(False)
            # else:
            #     is_unigram.append(True)

        self.cono_grounding = torch.tensor(
            id_to_cono, dtype=torch.float32, device=self.device).clamp(min=0)
        _, self.discrete_cono = self.cono_grounding.topk(1)
        # combined_freq = torch.tensor(
        #     id_to_freq, dtype=torch.int64, device=self.device).sum(dim=1)

        # gd = self.cono_grounding.clone()  # making a copy to be safe
        # # gd = F.normalize(gd, p=1)  # for freq ratio, not for PMI

        # # Zero out low frequency words
        # gd[combined_freq < 1000] = torch.zeros(3, device=self.device)
        # gd[is_unigram] = torch.zeros(3, device=self.device)

        # num_samples = 300
        # # 3 bins
        # # exclude the top PMI which is always UNK
        # _, self.liberal_ids = gd[:, 0].topk(num_samples)
        # _, self.neutral_ids = gd[:, 1].topk(num_samples)
        # _, self.conservative_ids = gd[:, 2].topk(num_samples)

        # # # 5 bins
        # _, self.socialist_ids = gd[:, 0].topk(num_samples)
        # _, self.liberal_ids = gd[:, 1].topk(num_samples)
        # _, self.neutral_ids = gd[:, 2].topk(num_samples)
        # _, self.conservative_ids = gd[:, 3].topk(num_samples)
        # _, self.chauvinist_ids = gd[:, 4].topk(num_samples)

        # # For debugging
        # # print('Socialist:')
        # # for i in self.socialist_ids:
        # #     print(ground[self.id_to_word[i.item()]])
        # print('\n\nLiberal:')
        # for i in self.liberal_ids:
        #     print(ground[self.id_to_word[i.item()]])
        # print('\n\nNeutral:')
        # for i in self.neutral_ids:
        #     print(ground[self.id_to_word[i.item()]])
        # print('\n\nConservative:')
        # for i in self.conservative_ids:
        #     print(ground[self.id_to_word[i.item()]])
        # # print('\n\nChauvinist:')
        # # for i in self.chauvinist_ids:
        # #     print(ground[self.id_to_word[i.item()]])
        # raise SystemExit

        # Initailize denotation grounding
        with open('../data/ellie/partisan_sample_val.hp.txt') as file:
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
        # self.deno_grounding.update(pretrained_neighbors(self.liberal_ids))
        # self.deno_grounding.update(pretrained_neighbors(self.neutral_ids))
        # self.deno_grounding.update(pretrained_neighbors(self.conservative_ids))
        # self.dev_ids = torch.cat(
        #     [self.liberal_ids, self.neutral_ids, self.conservative_ids])

    def forward(
            self,
            center_word_ids: Vector,
            true_context_ids: Vector,
            seq_word_ids: Matrix,
            cono_labels: Vector,
            recompose: bool = False
            ) -> Scalar:
        # Denotation Probe
        deno_loss = self.skip_gram_loss(center_word_ids, true_context_ids)

        # Connotation Probe
        seq_word_vecs: R3Tensor = self.embedding(seq_word_ids)
        seq_repr: Matrix = torch.mean(seq_word_vecs, dim=1)
        cono_logits = self.cono_decoder(seq_repr)
        cono_log_prob = F.log_softmax(cono_logits, dim=1)
        proper_cono_loss = F.nll_loss(cono_log_prob, cono_labels)

        if self.gamma < 0:  # DS removing connotation
            uniform_dist = torch.full_like(cono_log_prob, 1 / self.num_cono_classes)
            adversary_cono_loss = F.kl_div(cono_log_prob, uniform_dist, reduction='batchmean')
            decomposer_loss = torch.sigmoid(deno_loss) + torch.sigmoid(adversary_cono_loss)
        else:  # CS removing denotation
            decomposer_loss = (1 + self.delta * torch.sigmoid(deno_loss)
                               + self.gamma * torch.sigmoid(proper_cono_loss))
            adversary_cono_loss = proper_cono_loss

        if recompose:
            return decomposer_loss, deno_loss, proper_cono_loss, adversary_cono_loss, seq_word_vecs
        else:
            return decomposer_loss, deno_loss, proper_cono_loss, adversary_cono_loss

    def skip_gram_loss(
            self,
            center_word_ids: Vector,
            true_context_ids: Vector
            ) -> Scalar:
        # negative_context_ids = torch.multinomial(
        #     self.negative_sampling_probs,
        #     len(true_context_ids),
        #     replacement=True
        #     ).to(self.device)
        # center = self.embedding(center_word_ids)
        # true_context = self.embedding(true_context_ids)
        # negative_context = self.embedding(negative_context_ids)

        # true_context_loss = F.cosine_embedding_loss(
        #     center, true_context, torch.ones_like(center_word_ids))
        # negative_context_loss = F.cosine_embedding_loss(
        #     center, negative_context, torch.zeros_like(center_word_ids))
        # deno_loss = true_context_loss - negative_context_loss

        # Faster but less readable
        negative_context_ids = torch.multinomial(
            self.negative_sampling_probs,
            len(true_context_ids) * self.num_negative_samples,
            replacement=True
        ).view(len(true_context_ids), self.num_negative_samples).to(self.device)

        center = self.embedding(center_word_ids)
        true_context = self.embedding(true_context_ids)
        negative_context = self.embedding(negative_context_ids)

        # batch_size * embed_size
        objective = torch.sum(  # dot product
            torch.mul(center, true_context),  # Hadamard product
            dim=1)  # be -> b
        objective = F.logsigmoid(objective)

        # batch_size * num_negative_samples * embed_size
        # negative_context: bne
        # center: be -> be1
        negative_objective = torch.bmm(  # bne, be1 -> bn1
            negative_context, center.unsqueeze(2)
            ).squeeze()  # bn1 -> bn
        negative_objective = F.logsigmoid(-negative_objective)
        negative_objective = torch.sum(negative_objective, dim=1)  # bn -> b
        return -torch.mean(objective + negative_objective)

    def predict(self, seq_word_ids: Vector) -> Vector:
        self.eval()
        with torch.no_grad():
            word_vecs: R3Tensor = self.embedding(seq_word_ids)
            seq_repr: Matrix = torch.mean(word_vecs, dim=1)
            cono = self.cono_decoder(seq_repr)
            cono_conf = F.softmax(cono, dim=1)
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
            ) -> Matrix:
        with torch.no_grad():
            query_vectors = self.embedding(query_ids)
            try:
                cos_sim = F.cosine_similarity(
                    query_vectors.unsqueeze(1),
                    self.embedding.weight.unsqueeze(0),
                    dim=2)
            except RuntimeError:  # insufficient GPU memory
                cos_sim = torch.stack([
                    F.cosine_similarity(qv.unsqueeze(0), self.embedding.weight)
                    for qv in query_vectors])
            cos_sim, neighbor_ids = cos_sim.topk(k=top_k, dim=-1)
            if verbose:
                return cos_sim[:, 1:], neighbor_ids[:, 1:]
            else:  # excludes the first neighbor, which is always the query itself
                return neighbor_ids[:, 1:]

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

            # query_cono: Vector = self.cono_grounding[query_id]
            # try:
            #     neighbor_cono = torch.stack([
            #         self.cono_grounding[nid] for nid in neighbor_ids])
            #     diveregence = F.kl_div(
            #         query_cono.unsqueeze(0),
            #         neighbor_cono,
            #         reduction='batchmean').item()
            #     if np.isfinite(diveregence):
            #         cono_homogeneity.append(-diveregence)
            # except:
            #     pass

            query_cono_disc = self.discrete_cono[query_id]
            same_cono = len(
                [nid for nid in neighbor_ids
                 if self.discrete_cono[nid] == query_cono_disc])
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


class LabeledDocuments(torch.utils.data.IterableDataset):

    def __init__(self, config: 'DecomposerConfig'):
        super().__init__()
        self.batch_size = config.batch_size
        self.window_radius = config.skip_gram_window_radius
        self.numericalize_cono = config.numericalize_cono

        corpus_path = config.input_dir / 'train.pickle'
        print(f'Loading {corpus_path}', flush=True)
        with open(corpus_path, 'rb') as corpus_file:
            preprocessed = pickle.load(corpus_file)
        self.word_to_id: Dict[str, int] = preprocessed['word_to_id']
        self.id_to_word: Dict[int, str] = preprocessed['id_to_word']
        self.ground: Dict[str, GroundedWord] = preprocessed['ground']
        self.documents: List[LabeledDoc] = preprocessed['documents']
        self.negative_sampling_probs: Vector = torch.tensor(
            preprocessed['negative_sampling_probs'])
        assert preprocessed['numericalize_cono'] == config.numericalize_cono

        self.estimated_len = (
            sum([len(sent.numerical_tokens)
                 for doc in self.documents
                 for sent in doc.sentences])
            * self.window_radius // self.batch_size)

        # Set up multiprocessing
        self.total_workload = len(self.documents)
        self.worker_start: Optional[int] = None
        self.worker_end: Optional[int] = None

    # def __len__(self) -> int:  # print warnings when estimates are off
    #     return self.estimated_len

    def __iter__(self) -> Iterable[Tuple]:
        """
        Denotation: Parsing (center, context) word_id pairs
        Connotation: (a sentence of word_ids, cono_label)
        """
        documents = self.documents[self.worker_start:self.worker_end]
        random.shuffle(documents)
        batch_seq: List[Vector] = []
        batch_cono: List[int] = []
        batch_center: List[int] = []
        batch_context: List[int] = []
        for doc in documents:
            cono_label = self.numericalize_cono[doc.party]
            for sent in doc.sentences:
                if len(batch_center) > self.batch_size:
                    yield (
                        nn.utils.rnn.pad_sequence(batch_seq, batch_first=True),
                        torch.tensor(batch_center),
                        torch.tensor(batch_context),
                        torch.tensor(batch_cono))
                    batch_seq = []
                    batch_cono = []
                    batch_center = []
                    batch_context = []

                center_word_ids: List[int] = []
                context_word_ids: List[int] = []
                seq = sent.numerical_tokens
                for center_index, center_word_id in enumerate(seq):
                    left_index = max(center_index - self.window_radius, 0)
                    right_index = min(center_index + self.window_radius,
                                      len(seq) - 1)
                    context_word_id: List[int] = (
                        seq[left_index:center_index] +
                        seq[center_index + 1:right_index + 1])
                    context_word_ids += context_word_id
                    center_word_ids += [center_word_id] * len(context_word_id)
                # yield seq, center_word_ids, context_word_ids, cono_label  # regular batching
                batch_seq.append(torch.tensor(seq))
                batch_cono.append(cono_label)
                batch_center += center_word_ids
                batch_context += context_word_ids

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        worker = torch.utils.data.get_worker_info()
        assert worker.id == worker_id
        dataset = worker.dataset
        per_worker = dataset.total_workload // worker.num_workers
        dataset.worker_start = worker.id * per_worker
        dataset.worker_end = min(dataset.worker_start + per_worker,
                                 dataset.total_workload)
        # print(f'Worker {worker_id + 1}/{worker.num_workers} loading data '
        #       f'range({dataset.worker_start}, {dataset.worker_end})')


class DecomposerExperiment(Experiment):

    def __init__(self, config: 'DecomposerConfig'):
        super().__init__(config)
        self.data = LabeledDocuments(config)
        self.dataloader = torch.utils.data.DataLoader(
            self.data,
            # batch_size=config.batch_size,
            batch_size=None,  # disable auto batching, see __iter__
            # drop_last=True,
            # collate_fn=self.data.collate,
            num_workers=config.num_dataloader_threads,
            worker_init_fn=self.data.worker_init_fn,
            pin_memory=True)
        self.model = Decomposer(config, self.data)

        # For Debugging
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)  # param.data)

        # self.deno_params = self.model.deno_decoder.parameters()
        # self.deno_optimizer = config.optimizer(
        #     self.deno_params, lr=config.learning_rate)

        # self.cono_params = self.model.cono_decoder.parameters()
        self.cono_optimizer = config.optimizer(
            self.model.cono_decoder.parameters(), lr=config.learning_rate)

        # self.decomposer_params = self.model.embedding.parameters()
        self.decomposer_optimizer = config.optimizer(
            self.model.embedding.parameters(), lr=config.learning_rate)

        self.to_be_saved = {
            'config': self.config,
            'model': self.model}

    def train(self) -> None:
        config = self.config
        model = self.model
        # # For debugging
        # self.save_everything(self.config.output_dir / f'init.pt')
        # raise SystemExit

        # if config.auto_save_intra_epoch:
        #     save_per_batch = len(self.dataloader) // config.auto_save_intra_epoch
        # else:
        #     save_per_batch = None
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
                    total=self.data.estimated_len,
                    mininterval=config.progress_bar_refresh_rate,
                    desc='Batches')
            for batch_index, batch in batches:
                seq_word_ids = batch[0].to(self.device)
                center_word_ids = batch[1].to(self.device)
                context_word_ids = batch[2].to(self.device)
                cono_labels = batch[3].to(self.device)

                model.zero_grad()
                L_decomp, l_deno, l_cono_proper, l_cono_adversary = model(
                    center_word_ids, context_word_ids, seq_word_ids, cono_labels)
                L_decomp.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model.embedding.parameters(), config.clip_grad_norm)
                self.decomposer_optimizer.step()

                # model.zero_grad()
                # L_decomp, l_deno, l_cono_proper, l_cono_adversary = model(
                #     center_word_ids, context_word_ids, seq_word_ids, cono_labels)
                nn.utils.clip_grad_norm_(model.cono_decoder.parameters(), config.clip_grad_norm)
                l_cono_proper.backward()
                self.cono_optimizer.step()

                if batch_index % config.update_tensorboard == 0:
                    cono_accuracy = model.accuracy(seq_word_ids, cono_labels)
                    stats = {
                        'Decomposer/deno_loss': l_deno,
                        'Decomposer/cono_loss_proper': l_cono_proper,
                        'Decomposer/cono_loss_adversary': l_cono_adversary,
                        # 'Decomposer/accuracy_train_deno': deno_accuracy,
                        'Decomposer/accuracy_train_cono': cono_accuracy,
                        'Decomposer/combined_loss': L_decomp
                    }
                    self.update_tensorboard(stats)
                if batch_index % config.eval_dev_set == 0:
                    self.validation()
                # if config.print_stats and batch_index % config.print_stats == 0:
                #     self.print_stats(epoch_index, batch_index, stats)
                self.tb_global_step += 1
            # End Batches
            # self.print_timestamp(epoch_index)
            self.auto_save(epoch_index)
            self.data.estimated_len = batch_index + 1
        # End Epochs

    def validation(self) -> None:
        D_model = self.model
        DS_Hdeno, DS_Hcono = D_model.homemade_homogeneity(D_model.dev_ids)
        _, DS_Hcono_SP = D_model.SciPy_homogeneity(D_model.dev_ids)

        mean_delta, abs_rhos = word_sim.mean_delta(
            D_model.embedding.weight, D_model.pretrained_embed.weight,
            D_model.id_to_word, reduce=False)
        cos_sim = F.cosine_similarity(
            D_model.embedding.weight, D_model.pretrained_embed.weight).mean()
        self.update_tensorboard({
            'Decomposed Space/Neighbor Overlap': DS_Hdeno,
            'Decomposed Space/Party Homogeneity': DS_Hcono,
            'Decomposed Space/Party Homogeneity SciPy': DS_Hcono_SP,
            'Decomposed Space/Overlap - Party': DS_Hdeno - DS_Hcono,

            'Decomposed Space/rho difference cf pretrained': mean_delta,
            'Decomposed Space/MTurk-771': abs_rhos[0],
            'Decomposed Space/cosine cf pretrained': cos_sim
        })


@dataclass
class DecomposerConfig():
    # Essential
    input_dir: Path = Path('../data/ready/3bin')
    output_dir: Path = Path('../results/debug')
    device: torch.device = torch.device('cuda')
    debug_subset_corpus: Optional[int] = None
    # dev_holdout: int = 5_000
    # test_holdout: int = 10_000
    num_dataloader_threads: int = 0

    decomposed_size: int = 300
    delta: float = 1  # denotation classifier weight 𝛿
    gamma: float = -1  # connotation classifier weight 𝛾

    architecture: str = 'L4'
    dropout_p: float = 0.1
    batch_size: int = 1024
    embed_size: int = 300
    num_epochs: int = 10
    # encoder_update_cycle: int = 1  # per batch
    # decoder_update_cycle: int = 1  # per batch

    # pretrained_embedding: Optional[str] = None
    pretrained_embedding: Optional[Path] = Path('../data/pretrained_word2vec/partisan_news.txt')
    freeze_embedding: bool = False  # NOTE
    skip_gram_window_radius: int = 5
    num_negative_samples: int = 5
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
    auto_save_intra_epoch: Optional[int] = None
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
            '-sv', '--auto-save-per-epoch', action='store', type=int)
        parser.add_argument(
            '-pe', '--pretrained-embedding', action='store', type=Path)
        parser.parse_args(namespace=self)

        # self.numericalize_cono: Dict[str, int] = {
        #     'left': 0,
        #     'left-center': 1,
        #     'least': 2,
        #     'right-center': 3,
        #     'right': 4}
        self.numericalize_cono = {
            'left': 0,
            'left-center': 0,
            'least': 1,
            'right-center': 2,
            'right': 2}
        self.num_cono_classes = len(self.numericalize_cono)

        if self.architecture == 'linear':
            self.cono_decoder = nn.Linear(300, self.num_cono_classes)
        elif self.architecture == 'L1':
            self.cono_decoder = nn.Sequential(
                nn.Linear(300, self.num_cono_classes),
                nn.SELU())
        elif self.architecture == 'L2':
            self.cono_decoder = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, self.num_cono_classes),
                nn.SELU())
        elif self.architecture == 'L4':
            self.cono_decoder = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.AlphaDropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.SELU(),
                nn.AlphaDropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, self.num_cono_classes),
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
