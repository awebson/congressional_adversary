import argparse
import pickle
import random
from statistics import mean
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Iterable, Dict, Optional, Union

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
        self.cono_probe = cono_probe
        self.num_cono_classes = cono_probe[-1].out_features
        self.device = device
        self.to(self.device)

        # for skip-gram loss
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
        proper_cono_loss = F.nll_loss(cono_log_prob, cono_labels)

        deno_loss = self.skip_gram_loss(center_word_ids, true_context_ids)

        if self.preserve == 'deno':  # DS removing connotation (gamma < 0)
            uniform_dist = torch.full_like(cono_log_prob, 1 / self.num_cono_classes)
            adversary_cono_loss = F.kl_div(cono_log_prob, uniform_dist, reduction='batchmean')
            decomposer_loss = torch.sigmoid(deno_loss) + torch.sigmoid(adversary_cono_loss)
        else:  # CS removing denotation
            decomposer_loss = (1 - torch.sigmoid(deno_loss)
                               + torch.sigmoid(proper_cono_loss))
            adversary_cono_loss = 0  # placeholder

        return (decomposer_loss, deno_loss,
                proper_cono_loss, adversary_cono_loss, seq_word_vecs)

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
        # self.recomposer = nn.Linear(600, 300)
        self.rho = config.recomposer_rho
        self.to(self.device)

        self.id_to_word = data.id_to_word
        self.ground = data.ground

    def forward(
            self,
            center_word_ids: Vector,
            context_word_ids: Vector,
            seq_word_ids: Matrix,
            cono_labels: Vector,
            ) -> Tuple[Scalar, ...]:
        L_DS, DS_d, DS_cp, DS_ca, deno_vecs = self.deno_space(
            center_word_ids, context_word_ids, seq_word_ids, cono_labels)
        L_CS, CS_d, CS_cp, CS_ca, cono_vecs = self.cono_space(
            center_word_ids, context_word_ids, seq_word_ids, cono_labels)

        # recomposed = self.recomposer(torch.cat((deno_vecs, cono_vecs), dim=-1))
        recomposed = deno_vecs + cono_vecs  # cosine similarity ignores magnitude
        pretrained = self.pretrained_embed(seq_word_ids)
        L_R = 1 - F.cosine_similarity(recomposed, pretrained, dim=-1).mean()

        L_joint = L_DS + L_CS + self.rho * L_R
        return L_joint, L_R, L_DS, DS_d, DS_cp, DS_ca, L_CS, CS_d, CS_cp, CS_ca

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
        if next(iter(self.ground.values())).deno is not None:
            return super().homogeneity(query_ids, top_k)

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

        random.shuffle(self.documents)
        self.estimated_len = (
            sum([len(sent.numerical_tokens)
                 for doc in self.documents
                 for sent in doc.sentences])
            * self.window_radius // self.batch_size)

        with open(config.dev_path) as file:
            self.dev_ids = torch.tensor(
                [self.word_to_id[word.strip()] for word in file],
                device=config.device)
        with open(config.test_path) as file:
            self.test_ids = torch.tensor(
                [self.word_to_id[word.strip()] for word in file],
                device=config.device)
        with open(config.rand_path) as file:
            self.rand_ids = torch.tensor(
                [self.word_to_id[word.strip()] for word in file
                 if word.strip() in self.word_to_id],
                device=config.device)

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

        # self.D_decomp_params = model.deno_space.decomposed.parameters()
        # self.D_cono_params = model.deno_space.cono_probe.parameters()
        self.DS_decomp_optimizer = config.optimizer(
            model.deno_space.decomposed.parameters(), lr=config.learning_rate)
        self.DS_cono_optimizer = config.optimizer(
            model.deno_space.cono_probe.parameters(), lr=config.learning_rate)

        # self.C_decomp_params = model.cono_space.decomposed.parameters()
        # self.C_cono_parms = model.cono_space.cono_probe.parameters()
        self.CS_decomp_optimizer = config.optimizer(
            model.cono_space.decomposed.parameters(), lr=config.learning_rate)
        self.CS_cono_optimizer = config.optimizer(
            model.cono_space.cono_probe.parameters(), lr=config.learning_rate)

        # self.recomp_params = model.recomposer.parameters()
        # self.R_optimizer = config.optimizer(self.recomp_params, lr=config.learning_rate)

        dev_Hc = model.deno_space.homogeneity(self.data.dev_ids)
        test_Hc = model.deno_space.homogeneity(self.data.test_ids)
        rand_Hc = model.deno_space.homogeneity(self.data.rand_ids)
        model.PE_homogeneity = {
            'dev Hc': dev_Hc,
            'test Hc': test_Hc,
            'rand Hc': rand_Hc,
        }
        print(model.PE_homogeneity)

    def train_step(self, batch_index: int, batch: Tuple) -> None:
        model = self.model
        grad_clip = self.config.clip_grad_norm
        seq_word_ids = batch[0].to(self.device)
        center_word_ids = batch[1].to(self.device)
        context_word_ids = batch[2].to(self.device)
        cono_labels = batch[3].to(self.device)

        model.zero_grad()
        L_joint, L_R, L_DS, DS_d, DS_cp, DS_ca, L_CS, CS_d, CS_cp, CS_ca = model(
            center_word_ids, context_word_ids, seq_word_ids, cono_labels)
        L_joint.backward()
        nn.utils.clip_grad_norm_(model.deno_space.decomposed.parameters(), grad_clip)
        nn.utils.clip_grad_norm_(model.cono_space.decomposed.parameters(), grad_clip)
        self.DS_decomp_optimizer.step()
        self.CS_decomp_optimizer.step()

        # Update probes with proper losses
        model.zero_grad()
        L_DS, DS_d, DS_cp, DS_ca, _ = model.deno_space(
            center_word_ids, context_word_ids, seq_word_ids, cono_labels)
        DS_cp.backward()
        nn.utils.clip_grad_norm_(model.deno_space.cono_probe.parameters(), grad_clip)
        self.DS_cono_optimizer.step()

        # TODO already updated?
        # model.zero_grad()
        # L_CS, CS_d, CS_cp, CS_ca, _ = model.cono_space(
        #     center_word_ids, context_word_ids, seq_word_ids, cono_labels)
        # CS_cp.backward()
        # nn.utils.clip_grad_norm_(model.cono_space.cono_probe.parameters(), grad_clip)
        # self.CS_cono_optimizer.step()

        # # Recomposer
        # nn.utils.clip_grad_norm_(self.recomp_params, grad_clip)
        # self.R_optimizer.step()

        if batch_index % self.config.update_tensorboard == 0:
            DS_cono_acc, CS_cono_acc = model.accuracy(seq_word_ids, cono_labels)
            self.update_tensorboard({
                'Denotation Decomposer/deno_loss': DS_d,
                'Denotation Decomposer/cono_loss_proper': DS_cp,
                'Denotation Decomposer/cono_loss_adversary': DS_ca,
                'Denotation Decomposer/combined_loss': L_DS,
                'Denotation Decomposer/accuracy_train_cono': DS_cono_acc,

                'Connotation Decomposer/deno_loss': CS_d,
                'Connotation Decomposer/cono_loss_proper': CS_cp,
                'Connotation Decomposer/combined_loss': L_CS,
                'Connotation Decomposer/accuracy_train_cono': CS_cono_acc,

                'Joint/loss': L_joint,
                'Joint/Recomposer': L_R
            })
        if batch_index % self.config.eval_dev_set == 0:
            self.eval_step()

    def eval_step(self) -> None:
        PE = self.model.PE_homogeneity
        DS_Hc, CS_Hc = self.model.homogeneity(self.data.dev_ids)
        self.update_tensorboard({
            'Homogeneity Diff Dev/DS Hcono': DS_Hc - PE['dev Hc'],
            'Homogeneity Diff Dev/CS Hcono': CS_Hc - PE['dev Hc'],
        })
        DS_Hc, CS_Hc = self.model.homogeneity(self.data.test_ids)
        self.update_tensorboard({
            'Homogeneity Diff Test/DS Hcono': DS_Hc - PE['test Hc'],
            'Homogeneity Diff Test/CS Hcono': CS_Hc - PE['test Hc'],
        })
        DS_Hc, CS_Hc = self.model.homogeneity(self.data.rand_ids)
        self.update_tensorboard({
            'Homogeneity Diff Random/DS Hcono': DS_Hc - PE['rand Hc'],
            'Homogeneity Diff Random/CS Hcono': CS_Hc - PE['rand Hc'],
        })

        # model = self.model
        # D_model = model.deno_space
        # DS_Hdeno, DS_Hcono = D_model.homemade_homogeneity(D_model.dev_ids)
        # _, DS_Hcono_SP = D_model.SciPy_homogeneity(D_model.dev_ids)

        # mean_delta, abs_rhos = word_sim.mean_delta(
        #     D_model.decomposed.weight, D_model.pretrained_embed.weight,
        #     model.id_to_word, reduce=False)
        # cos_sim = F.cosine_similarity(
        #     D_model.decomposed.weight, D_model.pretrained_embed.weight).mean()
        # self.update_tensorboard({
        #     'Denotation Space/Neighbor Overlap': DS_Hdeno,
        #     'Denotation Space/Party Homogeneity': DS_Hcono,
        #     'Denotation Space/Party Homogeneity SciPy': DS_Hcono_SP,
        #     'Denotation Space/Overlap - Party': DS_Hdeno - DS_Hcono,

        #     'Denotation Space/rho difference cf pretrained': mean_delta,
        #     'Denotation Space/MTurk-771': abs_rhos[0],
        #     'Denotation Space/cosine cf pretrained': cos_sim
        # })

        # C_model = model.cono_space
        # CS_Hdeno, CS_Hcono = C_model.homemade_homogeneity(C_model.dev_ids)
        # _, CS_Hcono_SP = C_model.SciPy_homogeneity(C_model.dev_ids)
        # mean_delta, abs_rhos = word_sim.mean_delta(
        #     C_model.decomposed.weight, C_model.pretrained_embed.weight,
        #     model.id_to_word, reduce=False)
        # cos_sim = F.cosine_similarity(
        #     C_model.decomposed.weight, C_model.pretrained_embed.weight).mean()
        # self.update_tensorboard({
        #     'Connotation Space/Neighbor Overlap': CS_Hdeno,
        #     'Connotation Space/Party Homogeneity': CS_Hcono,
        #     'Connotation Space/Party Homogeneity SciPy': CS_Hcono_SP,
        #     'Connotation Space/Party - Overlap': CS_Hcono - CS_Hdeno,

        #     'Connotation Space/rho difference cf pretrained': mean_delta,
        #     'Connotation Space/MTurk-771': abs_rhos[0],
        #     'Connotation Space/cosine cf pretrained': cos_sim
        # })

        # with torch.no_grad():
        #     # sample = torch.randint(
        #     #     D_model.decomposed.num_embeddings, size=(25_000,), device=self.device)
        #     sample = torch.arange(D_model.decomposed.num_embeddings, device=self.device)
        #     recomposed = D_model.decomposed(sample) + C_model.decomposed(sample)

        # mean_delta, abs_rhos = word_sim.mean_delta(
        #     recomposed,
        #     D_model.pretrained_embed.weight,
        #     model.id_to_word,
        #     reduce=False)
        # self.update_tensorboard({
        #     'Recomposer/mean IntraSpace quality': ((DS_Hdeno - DS_Hcono) + (CS_Hcono - CS_Hdeno)) / 2,

        #     'Recomposer/rho difference cf pretrained': mean_delta,
        #     'Recomposer/MTurk-771': abs_rhos[0],
        #     'Recomposer/cosine similarity':
        #         F.cosine_similarity(recomposed, D_model.pretrained_embed(sample), dim=1).mean()
        # })


    def train(self) -> None:
        config = self.config
        # # For debugging
        # self.save_everything(self.config.output_dir / 'init_recomposer.pt')
        # raise SystemExit

        if not config.print_stats:
            epoch_pbar = tqdm(range(1, config.num_epochs + 1), desc=config.output_dir.name)
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
                self.train_step(batch_index, batch)
                self.tb_global_step += 1
            self.auto_save(epoch_index)

            self.data.estimated_len = batch_index
            if config.print_stats:
                self.print_timestamp(epoch_index)


@dataclass
class ProxyGroundedConfig():
    # Essential
    corpus_path: Path = Path('../../data/ready/CR_skip/train.pickle')
    numericalize_cono: Dict[str, int] = field(default_factory=lambda: {
        'D': 0,
        'R': 1})
    num_cono_classes: int = 2

    # corpus_path: Path = Path('../../data/ready/PN_skip')
    # numericalize_cono: Dict[str, int] = {
    #     'left': 0,
    #     'left-center': 0,
    #     'least': 1,
    #     'right-center': 2,
    #     'right': 2}
    # num_cono_classes: int = 3

    output_dir: Path = Path('../results/debug')
    device: torch.device = torch.device('cuda')

    rand_path: Path = Path('../../data/ellie/rand_sample.cr.txt')
    dev_path: Path = Path('../../data/ellie/partisan_sample_val.cr.txt')
    test_path: Path = Path('../../data/ellie/partisan_sample.cr.txt')
    pretrained_embed_path: Optional[Path] = Path(
        '../../data/pretrained_word2vec/for_real.txt')

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

    architecture: str = 'L4R'
    batch_size: int = 4096
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
    eval_dev_set: int = 100_000  # per batch  # NOTE
    progress_bar_refresh_rate: int = 1  # per second
    num_dataloader_threads: int = 0
    clear_tensorboard_log_in_output_dir: bool = True
    delete_all_exisiting_files_in_output_dir: bool = False
    auto_save_per_epoch: Optional[int] = 2
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
        elif self.architecture == 'L1':
            self.cono_probe = nn.Sequential(
                nn.Linear(300, self.num_cono_classes),
                nn.SELU())
        elif self.architecture == 'L2':
            self.cono_probe = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, self.num_cono_classes),
                nn.SELU())
        elif self.architecture == 'L4':
            self.cono_probe = nn.Sequential(
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
        elif self.architecture == 'L4R':
            self.cono_probe = nn.Sequential(
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
