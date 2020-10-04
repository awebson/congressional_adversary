import argparse
import pickle
import random
from statistics import mean
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Iterable, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import editdistance

from models.ideal_grounded import Decomposer, Recomposer
from data import Sentence, LabeledDoc, GroundedWord
from evaluations.word_similarity import all_wordsim as word_sim
from utils.experiment import Experiment
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor


class ProxyGroundedDecomposer(Decomposer):

    def __init__(
            self,
            preserve: str,  # either 'deno' or 'cono'
            initial_space: Matrix,
            cono_probe: nn.Module,
            id_to_word: Dict[int, str],
            ground: Dict[str, GroundedWord],
            num_negative_samples: int,
            negative_sampling_probs: Vector,
            device: torch.device):
        super(Decomposer, self).__init__()
        self.decomposed = nn.Embedding.from_pretrained(initial_space)
        self.decomposed.weight.requires_grad = True
        # self.SGNS_context = nn.Embedding.from_pretrained(initial_space)
        # self.SGNS_context.weight.requires_grad = True
        self.cono_probe = cono_probe
        self.num_cono_classes = cono_probe[-1].out_features
        self.device = device
        self.to(self.device)

        # for skip-gram negative sampling loss
        self.negative_sampling_probs = negative_sampling_probs
        self.num_negative_samples = num_negative_samples

        self.preserve = preserve
        self.id_to_word = id_to_word
        self.ground = ground

    def forward(
            self,
            center_word_ids: Vector,
            true_context_ids: Vector,
            seq_word_ids: Matrix,
            cono_labels: Vector,
            ) -> Tuple[Scalar, ...]:
        seq_word_vecs: R3Tensor = self.decomposed(seq_word_ids)
        seq_repr: Matrix = torch.mean(seq_word_vecs, dim=1)

        cono_logits = self.cono_probe(seq_repr)
        cono_log_prob = F.log_softmax(cono_logits, dim=1)
        cono_probe_loss = F.nll_loss(cono_log_prob, cono_labels)

        proxy_deno_loss = self.skip_gram_loss(center_word_ids, true_context_ids)

        if self.preserve == 'deno':  # DS removing connotation (gamma < 0)
            uniform_dist = torch.full_like(cono_log_prob, 1 / self.num_cono_classes)
            cono_adversary_loss = F.kl_div(cono_log_prob, uniform_dist, reduction='batchmean')
            return proxy_deno_loss, cono_probe_loss, cono_adversary_loss, seq_word_vecs
        else:  # CS removing denotation
            return proxy_deno_loss, cono_probe_loss, seq_word_vecs

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
        # center = self.decomposed(center_word_ids)
        # true_context = self.decomposed(true_context_ids)
        # negative_context = self.decomposed(negative_context_ids)

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

        center = self.decomposed(center_word_ids)
        true_context = self.decomposed(true_context_ids)
        negative_context = self.decomposed(negative_context_ids)
        # true_context = self.SGNS_context(true_context_ids)
        # negative_context = self.SGNS_context(negative_context_ids)

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
            word_vecs: R3Tensor = self.decomposed(seq_word_ids)
            seq_repr: Matrix = torch.mean(word_vecs, dim=1)
            cono = self.cono_probe(seq_repr)
            cono_conf = F.softmax(cono, dim=1)
        self.train()
        return cono_conf

    def accuracy(
            self,
            seq_word_ids: Matrix,
            cono_labels: Vector,
            ) -> float:
        cono_conf = self.predict(seq_word_ids)
        cono_predictions = cono_conf.argmax(dim=1)
        cono_correct_indicies = cono_predictions.eq(cono_labels)
        cono_accuracy = cono_correct_indicies.float().mean().item()
        return cono_accuracy

    def homogeneity(
            self,
            query_ids: Vector,
            top_k: int = 10
            ) -> float:
        # extra top_k buffer for excluding edit distance neighbors
        top_neighbor_ids = self.nearest_neighbors(query_ids, top_k + 5)
        cono_homogeneity = []
        for query_index, neighbor_ids in enumerate(top_neighbor_ids):
            query_id = query_ids[query_index].item()
            query_word = self.id_to_word[query_id]

            neighbor_ids = [
                nid for nid in neighbor_ids.tolist()
                if editdistance.eval(query_word, self.id_to_word[nid]) > 3]
            neighbor_ids = neighbor_ids[:top_k]

            if len(neighbor_ids) == 0:
                # print(query_word, [self.id_to_word[i.item()]
                #                    for i in top_neighbor_ids[query_index]])
                # raise RuntimeWarning
                continue

            query_cono = self.ground[query_word].majority_cono
            same_cono = 0
            for nid in neighbor_ids:
                neighbor_word = self.id_to_word[nid]
                try:
                    neighbor_cono = self.ground[neighbor_word].majority_cono
                    if neighbor_cono == query_cono:
                        same_cono += 1
                except KeyError:  # special tokens like [PAD] are ungrounded
                    continue
            cono_homogeneity.append(same_cono / len(neighbor_ids))

            # Continuous Connotation Version
            # neighbor_cono = torch.stack([
            #     self.cono_grounding[nid] for nid in neighbor_ids])
            # diveregence = F.kl_div(
            #     query_cono.unsqueeze(0),
            #     neighbor_cono,
            #     reduction='batchmean').item()
            # if np.isfinite(diveregence):
            #     cono_homogeneity.append(-diveregence)
        return mean(cono_homogeneity)

    def extra_homogeneity(
            self,
            query_ids: Vector,
            top_k: int = 10
            ) -> Tuple[float, float]:
        # extra 5 top-k for excluding edit distance neighbors
        top_neighbor_ids = self.nearest_neighbors(query_ids, top_k + 5)
        deno_homogeneity = []
        cono_homogeneity = []
        for query_index, neighbor_ids in enumerate(top_neighbor_ids):
            query_id = query_ids[query_index].item()
            query_word = self.id_to_word[query_id]

            neighbor_ids = [
                nid for nid in neighbor_ids.tolist()
                if editdistance.eval(query_word, self.id_to_word[nid]) > 3]
            neighbor_ids = neighbor_ids[:top_k]
            if len(neighbor_ids) == 0:
                # print(query_word, [self.id_to_word[i.item()]
                #                    for i in top_neighbor_ids[query_index]])
                # raise RuntimeWarning
                continue

            query_deno = self.ground[query_word].majority_deno
            query_cono = self.ground[query_word].majority_cono
            same_deno = 0
            same_cono = 0
            for nid in neighbor_ids:
                try:
                    neighbor_word = self.id_to_word[nid]
                    neighbor_deno = self.ground[neighbor_word].majority_deno
                    neighbor_cono = self.ground[neighbor_word].majority_cono
                    if neighbor_deno == query_deno:
                        same_deno += 1
                    if neighbor_cono == query_cono:
                        same_cono += 1
                except (KeyError, AttributeError):  # special tokens like [PAD] are ungrounded
                    # print(neighbor_word)
                    continue
            deno_homogeneity.append(same_deno / len(neighbor_ids))
            cono_homogeneity.append(same_cono / len(neighbor_ids))
        return mean(deno_homogeneity), mean(cono_homogeneity)


class ProxyGroundedRecomposer(Recomposer):

    def __init__(
            self,
            config: 'ProxyGroundedConfig',
            data: 'LabeledDocuments'):
        super(Recomposer, self).__init__()
        self.device = config.device

        self.pretrained_embed = Experiment.load_txt_embedding(
            config.pretrained_embed_path, data.word_to_id)
        self.pretrained_embed.weight.requires_grad = False

        self.deno_space = ProxyGroundedDecomposer(
            preserve='deno',
            initial_space=self.pretrained_embed.weight,
            cono_probe=config.cono_probe,
            id_to_word=data.id_to_word,
            ground=data.ground,
            num_negative_samples=config.num_negative_samples,
            negative_sampling_probs=data.negative_sampling_probs,
            device=self.device)

        self.cono_space = ProxyGroundedDecomposer(
            preserve='cono',
            initial_space=self.pretrained_embed.weight,
            cono_probe=config.cono_probe,
            id_to_word=data.id_to_word,
            ground=data.ground,
            num_negative_samples=config.num_negative_samples,
            negative_sampling_probs=data.negative_sampling_probs,
            device=self.device)

        # Recomposer
        self.recomposer = nn.Linear(600, 300)
        self.rho = config.recomposer_rho
        self.to(self.device)

        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.ground = data.ground
        self.eval_deno = hasattr(config, "extra_grounding")

    def forward(
            self,
            center_word_ids: Vector,
            context_word_ids: Vector,
            seq_word_ids: Matrix,
            cono_labels: Vector,
            ) -> Tuple[Scalar, ...]:
        # Denotation Space
        DS_deno_proxy, DS_cono_probe, DS_cono_adver, deno_vecs = self.deno_space(
            center_word_ids, context_word_ids, seq_word_ids, cono_labels)
        DS_decomp = torch.sigmoid(DS_deno_proxy) + torch.sigmoid(DS_cono_adver)

        # Connotation Space
        CS_deno_proxy, CS_cono_probe, cono_vecs = self.cono_space(
            center_word_ids, context_word_ids, seq_word_ids, cono_labels)
        CS_decomp = 1 - torch.sigmoid(CS_deno_proxy) + torch.sigmoid(CS_cono_probe)

        # Recomposer
        recomposed = self.recomposer(torch.cat((deno_vecs, cono_vecs), dim=-1))
        # recomposed = deno_vecs + cono_vecs  # cosine similarity ignores magnitude
        pretrained = self.pretrained_embed(seq_word_ids)
        L_R = 1 - F.cosine_similarity(recomposed, pretrained, dim=-1).mean()

        L_joint = DS_decomp + CS_decomp + self.rho * L_R
        return (L_joint, L_R,
                DS_decomp, DS_deno_proxy, DS_cono_probe, DS_cono_adver,
                CS_decomp, CS_deno_proxy, CS_cono_probe)

    def predict(self, seq_word_ids: Vector) -> Tuple[Vector, Vector]:
        DS_cono_conf = self.deno_space.predict(seq_word_ids)
        CS_cono_conf = self.cono_space.predict(seq_word_ids)
        return DS_cono_conf, CS_cono_conf

    def accuracy(
            self,
            seq_word_ids: Matrix,
            cono_labels: Vector,
            ) -> Tuple[float, float]:
        DS_cono_acc = self.deno_space.accuracy(seq_word_ids, cono_labels)
        CS_cono_acc = self.cono_space.accuracy(seq_word_ids, cono_labels)
        return DS_cono_acc, CS_cono_acc

    def homogeneity(
            self,
            query_ids: Vector,
            top_k: int = 10
            ) -> Tuple[float, ...]:
        """
        Only compute connotation homogeneity; but if denotation grounding
        is added retroactively, call the parent class's homogeneity() method,
        which returns DS_Hdeno, DS_Hcono, CS_Hdeno, CS_Hcono
        """
        if self.eval_deno:
            # DS_Hdeno, DS_Hcono = Decomposer.homogeneity(
            #     self.deno_space, query_ids, top_k=top_k)
            # CS_Hdeno, CS_Hcono = Decomposer.homogeneity(
            #     self.cono_space, query_ids, top_k=top_k)
            DS_Hdeno, DS_Hcono = self.deno_space.extra_homogeneity(query_ids, top_k=top_k)
            CS_Hdeno, CS_Hcono = self.cono_space.extra_homogeneity(query_ids, top_k=top_k)
            return DS_Hdeno, DS_Hcono, CS_Hdeno, CS_Hcono

        DS_Hcono = self.deno_space.homogeneity(query_ids, top_k=top_k)
        CS_Hcono = self.cono_space.homogeneity(query_ids, top_k=top_k)
        return DS_Hcono, CS_Hcono

    # def tabulate(
    #         self,
    #         query_ids: Vector,
    #         prefix: str = '',
    #         suffix: str = '',
    #         rounding: int = 4,
    #         top_k: int = 10
    #         ) -> Dict[str, float]:
    #     row = {}
    #     D_model = self.deno_space
    #     C_model = self.cono_space
    #     DS_Hdeno, DS_Hcono = D_model.homemade_homogeneity(query_ids, top_k=10)
    #     CS_Hdeno, CS_Hcono = C_model.homemade_homogeneity(query_ids, top_k=10)

    #     row['DS Hdeno'] = DS_Hdeno
    #     row['DS Hcono'] = DS_Hcono
    #     row['CS Hdeno'] = CS_Hdeno
    #     row['CS Hcono'] = CS_Hcono
    #     return {prefix + key + suffix: round(val, rounding)
    #             for key, val in row.items()}


class LabeledDocuments(torch.utils.data.IterableDataset):

    def __init__(self, config: 'ProxyGroundedConfig'):
        super().__init__()
        self.batch_size = config.batch_size
        self.window_radius = config.skip_gram_window_radius
        self.numericalize_cono = config.numericalize_cono

        print(f'Loading {config.corpus_path}', flush=True)
        with open(config.corpus_path, 'rb') as corpus_file:
            preprocessed = pickle.load(corpus_file)
        self.word_to_id: Dict[str, int] = preprocessed['word_to_id']
        self.id_to_word: Dict[int, str] = preprocessed['id_to_word']
        self.ground: Dict[str, GroundedWord] = preprocessed['ground']
        self.documents: List[LabeledDoc] = preprocessed['documents']
        self.negative_sampling_probs: Vector = torch.tensor(
            preprocessed['negative_sampling_probs'])

        if hasattr(config, "extra_grounding"):
            with open(config.extra_grounding, 'rb') as extra_file:
                preprocessed = pickle.load(extra_file)
            deno_ground = preprocessed['ground']
            for word in deno_ground.values():
                if word.text in self.ground:
                    self.ground[word.text].deno = word.deno
                    self.ground[word.text].majority_deno = word.majority_deno

        random.shuffle(self.documents)
        self.estimated_len = (
            sum([len(sent.numerical_tokens)
                 for doc in self.documents
                 for sent in doc.sentences])
            * self.window_radius // self.batch_size)

        def load_eval_words(path: Path) -> Vector:
            in_vocab = []
            with open(path) as file:
                for line in file:
                    word = line.strip()
                    if word in self.word_to_id:
                        in_vocab.append(word)
            if hasattr(config, "extra_grounding"):
                in_vocab = [w for w in in_vocab if w in deno_ground]
            print(f'Loaded {len(in_vocab)} in-vocab eval words from {path}')
            return torch.tensor(
                [self.word_to_id[w] for w in in_vocab], device=config.device)

        self.dev_ids = load_eval_words(config.dev_path)
        self.test_ids = load_eval_words(config.test_path)
        self.rand_ids = load_eval_words(config.rand_path)

        # Set up multiprocessing
        self.total_workload = len(self.documents)
        self.worker_start: Optional[int] = None
        self.worker_end: Optional[int] = None

    # def __len__(self) -> int:  # throws warnings when estimates are off
    #     return self.estimated_len

    def __iter__(self) -> Iterable[Tuple]:
        """
        Denotation: Parsing (center, context) word_id pairs
        Connotation: (a sentence of word_ids, cono_label)
        """
        documents = self.documents[self.worker_start:self.worker_end]
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


class ProxyGroundedExperiment(Experiment):

    def __init__(self, config: 'ProxyGroundedConfig'):
        super().__init__(config)
        self.data = LabeledDocuments(config)
        self.dataloader = torch.utils.data.DataLoader(
            self.data,
            batch_size=None,  # disable auto batching, see __iter__
            num_workers=config.num_dataloader_threads,
            worker_init_fn=self.data.worker_init_fn,
            pin_memory=True)
        self.model = ProxyGroundedRecomposer(config, self.data)
        model = self.model

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)  # param.data)

        self.DS_cono_optimizer = config.optimizer(
            model.deno_space.cono_probe.parameters(), lr=config.learning_rate)
        self.CS_cono_optimizer = config.optimizer(
            model.cono_space.cono_probe.parameters(), lr=config.learning_rate)

        self.joint_optimizer = config.optimizer(
            list(model.deno_space.decomposed.parameters()) +
            list(model.cono_space.decomposed.parameters()),
            lr=config.learning_rate)
        self.R_optimizer = config.optimizer(
            model.recomposer.parameters(), lr=config.learning_rate)

        model.dev_ids = self.data.dev_ids
        model.test_ids = self.data.test_ids
        model.rand_ids = self.data.rand_ids
        if not model.eval_deno:
            dev_Hc = model.deno_space.homogeneity(model.dev_ids)
            test_Hc = model.deno_space.homogeneity(model.test_ids)
            rand_Hc = model.deno_space.homogeneity(model.rand_ids)
            model.PE_homogeneity = {
                'dev Hc': dev_Hc,
                'test Hc': test_Hc,
                'rand Hc': rand_Hc,
            }
        else:
            dev_Hd, dev_Hc = model.deno_space.extra_homogeneity(model.dev_ids)
            test_Hd, test_Hc = model.deno_space.extra_homogeneity(model.test_ids)
            rand_Hd, rand_Hc = model.deno_space.extra_homogeneity(model.rand_ids)
            model.PE_homogeneity = {
                'dev Hd': dev_Hd,
                'dev Hc': dev_Hc,
                'test Hd': test_Hd,
                'test Hc': test_Hc,
                'rand Hd': rand_Hd,
                'rand Hc': rand_Hc,
            }
        print(model.PE_homogeneity)

    def train_step(self, batch_index: int, batch: Tuple) -> None:
        model = self.model
        seq_word_ids = batch[0].to(self.device)
        center_word_ids = batch[1].to(self.device)
        context_word_ids = batch[2].to(self.device)
        cono_labels = batch[3].to(self.device)

        # Update probes with proper (non-adversarial) losses
        model.zero_grad()
        DS_deno_proxy, DS_cono_probe, DS_cono_adver, _ = model.deno_space(
            center_word_ids, context_word_ids, seq_word_ids, cono_labels)
        DS_cono_probe.backward()
        self.DS_cono_optimizer.step()

        model.zero_grad()
        CS_deno_proxy, CS_cono_probe, _ = model.cono_space(
            center_word_ids, context_word_ids, seq_word_ids, cono_labels)
        CS_cono_probe.backward()
        self.CS_cono_optimizer.step()

        model.zero_grad()
        (L_joint, L_R,
            DS_decomp, DS_deno_proxy, DS_cono_probe, DS_cono_adver,
            CS_decomp, CS_deno_proxy, CS_cono_probe) = model(
            center_word_ids, context_word_ids, seq_word_ids, cono_labels)
        L_joint.backward()
        self.joint_optimizer.step()
        self.R_optimizer.step()

        if batch_index % self.config.update_tensorboard == 0:
            DS_cono_acc, CS_cono_acc = model.accuracy(seq_word_ids, cono_labels)
            self.update_tensorboard({
                'Denotation Decomposer/deno_loss': DS_deno_proxy,
                'Denotation Decomposer/cono_loss_proper': DS_cono_probe,
                'Denotation Decomposer/cono_loss_adversary': DS_cono_adver,
                'Denotation Decomposer/combined_loss': DS_decomp,
                'Denotation Decomposer/accuracy_train_cono': DS_cono_acc,

                'Connotation Decomposer/deno_loss': CS_cono_probe,
                'Connotation Decomposer/cono_loss_proper': CS_cono_probe,
                'Connotation Decomposer/combined_loss': CS_decomp,
                'Connotation Decomposer/accuracy_train_cono': CS_cono_acc,

                'Recomposer/joint loss': L_joint,
                'Recomposer/recomposition loss': L_R
            })
        # if batch_index % self.config.eval_dev_set == 0:
        #     self.eval_step()

    def eval_step(self, epoch_index: int) -> None:
        model = self.model
        PE = model.PE_homogeneity
        if not model.eval_deno:
            DS_Hc, CS_Hc = model.homogeneity(self.data.dev_ids)
            self.update_tensorboard({
                'Homogeneity Diff Dev/DS Hcono': DS_Hc - PE['dev Hc'],
                'Homogeneity Diff Dev/CS Hcono': CS_Hc - PE['dev Hc'],
                }, manual_step=epoch_index)
            DS_Hc, CS_Hc = model.homogeneity(self.data.test_ids)
            self.update_tensorboard({
                'Homogeneity Diff Test/DS Hcono': DS_Hc - PE['test Hc'],
                'Homogeneity Diff Test/CS Hcono': CS_Hc - PE['test Hc'],
                }, manual_step=epoch_index)
            DS_Hc, CS_Hc = model.homogeneity(self.data.rand_ids)
            self.update_tensorboard({
                'Homogeneity Diff Random/DS Hcono': DS_Hc - PE['rand Hc'],
                'Homogeneity Diff Random/CS Hcono': CS_Hc - PE['rand Hc'],
                }, manual_step=epoch_index)
        else:
            DS_Hd, DS_Hc, CS_Hd, CS_Hc = model.homogeneity(self.data.dev_ids)
            self.update_tensorboard({
                'Homogeneity Diff Dev/DS Hdeno': DS_Hd - PE['dev Hd'],
                'Homogeneity Diff Dev/DS Hcono': DS_Hc - PE['dev Hc'],
                'Homogeneity Diff Dev/CS Hdeno': CS_Hd - PE['dev Hd'],
                'Homogeneity Diff Dev/CS Hcono': CS_Hc - PE['dev Hc'],
                }, manual_step=epoch_index)
            DS_Hd, DS_Hc, CS_Hd, CS_Hc = model.homogeneity(self.data.test_ids)
            self.update_tensorboard({
                'Homogeneity Diff Test/DS Hdeno': DS_Hd - PE['test Hd'],
                'Homogeneity Diff Test/DS Hcono': DS_Hc - PE['test Hc'],
                'Homogeneity Diff Test/CS Hdeno': CS_Hd - PE['test Hd'],
                'Homogeneity Diff Test/CS Hcono': CS_Hc - PE['test Hc'],
                }, manual_step=epoch_index)
            DS_Hd, DS_Hc, CS_Hd, CS_Hc = model.homogeneity(self.data.rand_ids)
            self.update_tensorboard({
                'Homogeneity Diff Random/DS Hdeno': DS_Hd - PE['rand Hd'],
                'Homogeneity Diff Random/DS Hcono': DS_Hc - PE['rand Hc'],
                'Homogeneity Diff Random/CS Hdeno': CS_Hd - PE['rand Hd'],
                'Homogeneity Diff Random/CS Hcono': CS_Hc - PE['rand Hc'],
                }, manual_step=epoch_index)

        mean_delta, abs_rhos = word_sim.mean_delta(
            model.deno_space.decomposed.weight, model.pretrained_embed.weight,
            model.id_to_word, reduce=False)
        cos_sim = F.cosine_similarity(
            model.deno_space.decomposed.weight, model.pretrained_embed.weight).mean()
        self.update_tensorboard({
            # 'Denotation Decomposer/rho difference cf pretrained': mean_delta,
            'Denotation Decomposer/MTurk-771': abs_rhos[0],
            'Denotation Decomposer/cosine similarity': cos_sim
            }, manual_step=epoch_index)

        mean_delta, abs_rhos = word_sim.mean_delta(
            model.cono_space.decomposed.weight, model.pretrained_embed.weight,
            model.id_to_word, reduce=False)
        cos_sim = F.cosine_similarity(
            model.cono_space.decomposed.weight, model.pretrained_embed.weight).mean()
        self.update_tensorboard({
            'Connotation Decomposer/MTurk-771': abs_rhos[0],
            'Connotation Decomposer/cosine similarity': cos_sim
            }, manual_step=epoch_index)

        with torch.no_grad():
            # sample = torch.randint(
            #     D_model.decomposed.num_embeddings, size=(25_000,), device=self.device)
            sample = torch.arange(model.pretrained_embed.num_embeddings, device=self.device)
            # recomposed = model.deno_space.decomposed(sample) + model.cono_space.decomposed(sample)
            deno_vecs = model.deno_space.decomposed(sample)
            cono_vecs = model.cono_space.decomposed(sample)
            recomposed = model.recomposer(torch.cat((deno_vecs, cono_vecs), dim=-1))

        mean_delta, abs_rhos = word_sim.mean_delta(
            recomposed, model.pretrained_embed.weight,
            model.id_to_word, reduce=False)
        cos_sim = F.cosine_similarity(recomposed, model.pretrained_embed.weight).mean()
        self.update_tensorboard({
            # 'Recomposer/rho difference cf pretrained': mean_delta,
            'Recomposer/MTurk-771': abs_rhos[0],
            'Recomposer/cosine similarity': cos_sim
            }, manual_step=epoch_index)

    def train(self) -> None:
        config = self.config
        # # For debugging
        # self.save_everything(self.config.output_dir / 'init_recomposer.pt')
        # raise SystemExit

        if not config.print_stats:
            epoch_pbar = tqdm(range(1, config.num_epochs + 1), desc=config.output_dir.name)
        else:
            epoch_pbar = tqdm(range(1, config.num_epochs + 1), desc='Epochs')

        # estimated_save = self.data.estimated_len // 10

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
                self.train_step(batch_index, batch)
                self.tb_global_step += 1
                # # For very long epochs
                # if batch_index % estimated_save == 0:
                #     point = batch_index // estimated_save
                #     self.save_everything(
                #         self.config.output_dir / f'epoch{epoch_index}_{point}.pt')
            self.auto_save(epoch_index)
            self.eval_step(epoch_index)
            self.data.estimated_len = batch_index
            if config.print_stats:
                self.print_timestamp(epoch_index)


@dataclass
class ProxyGroundedConfig():
    # Congressional Record
    corpus_path: Path = Path('../../data/ready/CR_proxy/train.pickle')
    numericalize_cono: Dict[str, int] = field(default_factory=lambda: {
        'D': 0,
        'R': 1})
    num_cono_classes: int = 2
    rand_path: Path = Path('../../data/ready/CR_proxy/eval_words_random.txt')
    dev_path: Path = Path('../../data/ready/CR_proxy/0.7partisan_dev_words.txt')
    test_path: Path = Path('../../data/ready/CR_proxy/0.7partisan_test_words.txt')
    pretrained_embed_path: Optional[Path] = Path(
        '../../data/pretrained_word2vec/CR_proxy.txt')
    extra_grounding: Path = Path(
        '../../data/ready/CR_bill_context3/train_data.pickle')

    # # Partisan News
    # corpus_path: Path = Path('../../data/ready/PN_proxy/train.pickle')
    # numericalize_cono: Dict[str, int] = field(default_factory=lambda: {
    #     'left': 0,
    #     'left-center': 0,
    #     'least': 1,
    #     'right-center': 2,
    #     'right': 2})
    # num_cono_classes: int = 3
    # rand_path: Path = Path('../../data/ready/PN_proxy/eval_words_random.txt')
    # dev_path: Path = Path('../../data/ready/PN_proxy/0.7partisan_dev_words.txt')
    # test_path: Path = Path('../../data/ready/PN_proxy/0.7partisan_test_words.txt')
    # pretrained_embed_path: Optional[Path] = Path(
    #     '../../data/pretrained_word2vec/PN_proxy.txt')

    output_dir: Path = Path('../results/debug')
    device: torch.device = torch.device('cuda')

    # Denotation Decomposer
    deno_size: int = 300
    # deno_delta: float = 1  # denotation weight 𝛿
    # deno_gamma: float = -1  # connotation weight 𝛾

    # Conotation Decomposer
    cono_size: int = 300
    # cono_delta: float = -1  # denotation weight 𝛿
    # cono_gamma: float = 1  # connotation weight 𝛾

    # Recomposer
    recomposer_rho: float = 1
    dropout_p: float = 0.33

    architecture: str = 'MLP4'
    batch_size: int = 8192
    embed_size: int = 300
    num_epochs: int = 30

    skip_gram_window_radius: int = 5
    num_negative_samples: int = 10
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    learning_rate: float = 1e-4
    # momentum: float = 0.5
    # lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    # num_prediction_classes: int = 5
    clip_grad_norm: float = 10.0

    # Housekeeping
    # export_error_analysis: Optional[int] = 1  # per epoch
    update_tensorboard: int = 1000  # per batch
    print_stats: Optional[int] = 10_000  # per batch
    eval_dev_set: int = 10_000  # per batch  # NOTE
    progress_bar_refresh_rate: int = 1  # per second
    num_dataloader_threads: int = 0
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
            '-d', '--device', action='store', type=str)

        parser.add_argument(
            '-a', '--architecture', action='store', type=str)
        # parser.add_argument(
        #     '-dd', '--deno-delta', action='store', type=float)
        # parser.add_argument(
        #     '-dg', '--deno-gamma', action='store', type=float)
        # parser.add_argument(
        #     '-cd', '--cono-delta', action='store', type=float)
        # parser.add_argument(
        #     '-cg', '--cono-gamma', action='store', type=float)
        parser.add_argument(
            '-ns', '--num-negative-samples', action='store', type=int)

        parser.add_argument(
            '-lr', '--learning-rate', action='store', type=float)
        parser.add_argument(
            '-bs', '--batch-size', action='store', type=int)
        parser.add_argument(
            '-ep', '--num-epochs', action='store', type=int)
        parser.add_argument(
            '-sv', '--auto-save-per-epoch', action='store', type=int)
        parser.parse_args(namespace=self)

        assert self.num_cono_classes == len(set(self.numericalize_cono.values()))

        if self.architecture == 'linear':
            self.cono_probe = nn.Linear(300, self.num_cono_classes)
        elif self.architecture == 'MLP1':
            self.cono_probe = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, self.num_cono_classes))
        elif self.architecture == 'MLP2':
            self.cono_probe = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, self.num_cono_classes),
                nn.ReLU())
        elif self.architecture == 'MLP4':
            self.cono_probe = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, self.num_cono_classes))
        else:
            raise ValueError('Unknown architecture argument.')

        assert self.cono_probe[-1].out_features == self.num_cono_classes


def main() -> None:
    config = ProxyGroundedConfig()
    black_box = ProxyGroundedExperiment(config)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
