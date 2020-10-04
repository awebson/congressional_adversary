import argparse
import pickle
from statistics import mean
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn
from tqdm import tqdm
import editdistance

from data import GroundedWord
from utils.experiment import Experiment
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor


class Decomposer(nn.Module):

    def __init__(
            self,
            preserve: str,  # either 'deno' or 'cono'
            initial_space: Matrix,
            deno_probe: nn.Module,
            cono_probe: nn.Module,
            id_to_word: Dict[int, str],
            ground: Dict[str, GroundedWord],
            device: torch.device):
        """
        Denotation Loss: bill title or policy topic classifier
        Connotation Loss: party classifier

        If preserve = 'deno', decomposer will preserve deno and remove cono
        information from the decomposed space, and vice versa.
        """
        super().__init__()
        self.decomposed = nn.Embedding.from_pretrained(initial_space)
        self.decomposed.weight.requires_grad = True
        self.deno_probe = deno_probe
        self.cono_probe = cono_probe
        self.num_deno_classes = deno_probe[-1].out_features
        self.num_cono_classes = cono_probe[-1].out_features
        self.device = device
        self.to(self.device)

        self.preserve = preserve
        # self.deno_to_id = data.deno_to_id  # for homogeneity evaluation
        # self.id_to_deno = data.id_to_deno  # for error analysis
        # self.word_to_id = data.word_to_id
        self.id_to_word = id_to_word
        self.ground = ground

    def forward(
            self,
            seq_word_ids: Matrix,
            deno_labels: Vector,
            cono_labels: Vector
            ) -> Tuple[Scalar, ...]:
        seq_word_vecs: R3Tensor = self.decomposed(seq_word_ids)
        seq_repr: Matrix = torch.mean(seq_word_vecs, dim=1)

        deno_logits = self.deno_probe(seq_repr)
        deno_log_prob = F.log_softmax(deno_logits, dim=1)
        deno_probe_loss = F.nll_loss(deno_log_prob, deno_labels)

        cono_logits = self.cono_probe(seq_repr)
        cono_log_prob = F.log_softmax(cono_logits, dim=1)
        cono_probe_loss = F.nll_loss(cono_log_prob, cono_labels)

        if self.preserve == 'deno':  # DS removing connotation (gamma < 0)
            uniform_dist = torch.full_like(cono_log_prob, 1 / self.num_cono_classes)
            cono_adversary_loss = F.kl_div(cono_log_prob, uniform_dist, reduction='batchmean')
            return deno_probe_loss, cono_probe_loss, cono_adversary_loss, seq_word_vecs
        else:  # CS removing denotation
            uniform_dist = torch.full_like(deno_log_prob, 1 / self.num_deno_classes)
            deno_adversary_loss = F.kl_div(deno_log_prob, uniform_dist, reduction='batchmean')
            return deno_probe_loss, deno_adversary_loss, cono_probe_loss, seq_word_vecs

    def predict(self, seq_word_ids: Vector) -> Vector:
        self.eval()
        with torch.no_grad():
            word_vecs: R3Tensor = self.decomposed(seq_word_ids)
            seq_repr: Matrix = torch.mean(word_vecs, dim=1)
            deno = self.deno_probe(seq_repr)
            cono = self.cono_probe(seq_repr)
            deno_conf = F.softmax(deno, dim=1)
            cono_conf = F.softmax(cono, dim=1)
        self.train()
        return deno_conf, cono_conf

    def accuracy(
            self,
            seq_word_ids: Matrix,
            deno_labels: Vector,
            cono_labels: Vector,
            error_analysis_path: Optional[str] = None
            ) -> Tuple[float, float]:
        deno_conf, cono_conf = self.predict(seq_word_ids)
        deno_predictions = deno_conf.argmax(dim=1)
        cono_predictions = cono_conf.argmax(dim=1)
        # # Random Guess Baseline
        # deno_predictions = torch.randint_like(deno_labels, high=len(self.deno_to_id))
        # # Majority Class Baseline
        # majority_label = self.deno_to_id['Health']
        # deno_predictions = torch.full_like(deno_labels, majority_label)

        deno_correct_indicies = deno_predictions.eq(deno_labels)
        cono_correct_indicies = cono_predictions.eq(cono_labels)
        deno_accuracy = deno_correct_indicies.float().mean().item()
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
        return deno_accuracy, cono_accuracy

    def nearest_neighbors(
            self,
            query_ids: Vector,
            top_k: int = 10,
            verbose: bool = False,
            ) -> Matrix:
        with torch.no_grad():
            query_vectors = self.decomposed(query_ids)
            try:
                cos_sim = F.cosine_similarity(
                    query_vectors.unsqueeze(1),
                    self.decomposed.weight.unsqueeze(0),
                    dim=2)
            except RuntimeError:  # insufficient GPU memory
                cos_sim = torch.stack([
                    F.cosine_similarity(qv.unsqueeze(0), self.decomposed.weight)
                    for qv in query_vectors])
            cos_sim, neighbor_ids = cos_sim.topk(k=top_k, dim=-1)
            if verbose:
                return cos_sim[:, 1:], neighbor_ids[:, 1:]
            else:  # excludes the first neighbor, which is always the query itself
                return neighbor_ids[:, 1:]

    def homogeneity(
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
                except KeyError:  # special tokens like [PAD] are ungrounded
                    continue
            deno_homogeneity.append(same_deno / len(neighbor_ids))
            cono_homogeneity.append(same_cono / len(neighbor_ids))
        return mean(deno_homogeneity), mean(cono_homogeneity)


class Recomposer(nn.Module):

    def __init__(
            self,
            config: 'IdealGroundedConfig',
            data: 'LabeledSentences'):
        super().__init__()
        self.device = config.device

        self.pretrained_embed = Experiment.load_txt_embedding(
            config.pretrained_embed_path, data.word_to_id)
        self.pretrained_embed.weight.requires_grad = False

        self.deno_space = Decomposer(
            preserve='deno',
            initial_space=self.pretrained_embed.weight,
            deno_probe=config.deno_probe,
            cono_probe=config.cono_probe,
            id_to_word=data.id_to_word,
            ground=data.ground,
            device=self.device)

        self.cono_space = Decomposer(
            preserve='cono',
            initial_space=self.pretrained_embed.weight,
            deno_probe=config.deno_probe,
            cono_probe=config.cono_probe,
            id_to_word=data.id_to_word,
            ground=data.ground,
            device=self.device)

        # Recomposer
        self.recomposer = nn.Linear(600, 300)
        self.rho = config.recomposer_rho
        self.to(self.device)

        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.ground = data.ground

    def forward(
            self,
            seq_word_ids: Matrix,
            deno_labels: Vector,
            cono_labels: Vector
            ) -> Tuple[Scalar, ...]:
        # Denotation Space
        DS_deno_probe, DS_cono_probe, DS_cono_adver, deno_vecs = self.deno_space(
            seq_word_ids, deno_labels, cono_labels)
        DS_decomp = torch.sigmoid(DS_deno_probe) + torch.sigmoid(DS_cono_adver)

        # Connotation Space
        CS_deno_probe, CS_deno_adver, CS_cono_probe, cono_vecs = self.cono_space(
            seq_word_ids, deno_labels, cono_labels)
        CS_decomp = torch.sigmoid(CS_deno_adver) + torch.sigmoid(CS_cono_probe)

        # Recomposed Space
        recomposed = self.recomposer(torch.cat((deno_vecs, cono_vecs), dim=-1))
        # recomposed = deno_vecs + cono_vecs  # cosine similarity ignores magnitude
        pretrained = self.pretrained_embed(seq_word_ids)
        L_R = 1 - F.cosine_similarity(recomposed, pretrained, dim=-1).mean()

        L_joint = DS_decomp + CS_decomp + self.rho * L_R
        return (L_joint, L_R,
                DS_decomp, DS_deno_probe, DS_cono_probe, DS_cono_adver,
                CS_decomp, CS_deno_probe, CS_deno_adver, CS_cono_probe)

    def predict(self, seq_word_ids: Vector) -> Tuple[Vector, ...]:
        DS_deno_conf, DS_cono_conf = self.deno_space.predict(seq_word_ids)
        CS_deno_conf, CS_cono_conf = self.cono_space.predict(seq_word_ids)
        return DS_deno_conf, DS_cono_conf, CS_deno_conf, CS_cono_conf

    def accuracy(
            self,
            seq_word_ids: Matrix,
            deno_labels: Vector,
            cono_labels: Vector,
            error_analysis_path: Optional[str] = None
            ) -> Tuple[float, ...]:
        DS_deno_acc, DS_cono_acc = self.deno_space.accuracy(
            seq_word_ids, deno_labels, cono_labels)
        CS_deno_acc, CS_cono_acc = self.cono_space.accuracy(
            seq_word_ids, deno_labels, cono_labels)
        return DS_deno_acc, DS_cono_acc, CS_deno_acc, CS_cono_acc

    def homogeneity(
            self,
            query_ids: Vector,
            top_k: int = 10
            ) -> Tuple[float, ...]:
        DS_Hdeno, DS_Hcono = self.deno_space.homogeneity(query_ids, top_k=top_k)
        CS_Hdeno, CS_Hcono = self.cono_space.homogeneity(query_ids, top_k=top_k)
        return DS_Hdeno, DS_Hcono, CS_Hdeno, CS_Hcono

    def tabulate(
            self,
            # dev_ids: Vector,
            # test_ids: Vector,
            # rand_ids: Vector,
            rounding: int = 4,
            top_k: int = 10
            ) -> Dict[str, float]:
        row = {}
        PE = self.PE_homogeneity
        DS_Hd, DS_Hc, CS_Hd, CS_Hc = self.homogeneity(self.dev_ids)
        row.update({
            'Dev DS Hdeno': DS_Hd,
            'Dev DS Hcono': DS_Hc,
            'Dev CS Hdeno': CS_Hd,
            'Dev CS Hcono': CS_Hc,
            'Dev DS Hdeno delta': DS_Hd - PE['dev Hd'],
            'Dev DS Hcono delta': DS_Hc - PE['dev Hc'],
            'Dev CS Hdeno delta': CS_Hd - PE['dev Hd'],
            'Dev CS Hcono delta': CS_Hc - PE['dev Hc'],
        })
        DS_Hd, DS_Hc, CS_Hd, CS_Hc = self.homogeneity(self.test_ids)
        row.update({
            'Test DS Hdeno': DS_Hd,
            'Test DS Hcono': DS_Hc,
            'Test CS Hdeno': CS_Hd,
            'Test CS Hcono': CS_Hc,
            'Test DS Hdeno delta': DS_Hd - PE['test Hd'],
            'Test DS Hcono delta': DS_Hc - PE['test Hc'],
            'Test CS Hdeno delta': CS_Hd - PE['test Hd'],
            'Test CS Hcono delta': CS_Hc - PE['test Hc'],
        })
        DS_Hd, DS_Hc, CS_Hd, CS_Hc = self.homogeneity(self.rand_ids)
        row.update({
            'Random DS Hdeno': DS_Hd,
            'Random DS Hcono': DS_Hc,
            'Random CS Hdeno': CS_Hd,
            'Random CS Hcono': CS_Hc,
            'Random DS Hdeno delta': DS_Hd - PE['rand Hd'],
            'Random DS Hcono delta': DS_Hc - PE['rand Hc'],
            'Random CS Hdeno delta': CS_Hd - PE['rand Hd'],
            'Random CS Hcono delta': CS_Hc - PE['rand Hc'],
        })
        return {key: round(val, rounding) for key, val in row.items()}

    def cf_cos_sim(self, query1: str, query2: str) -> Tuple[float, ...]:
        try:
            query1_id = torch.tensor(self.word_to_id[query1], device=self.device)
        except KeyError:
            print(f'Out of vocabulary: {query1}')
            return -1, -1, -1
        try:
            query2_id = torch.tensor(self.word_to_id[query2], device=self.device)
        except KeyError:
            print(f'Out of vocabulary: {query2}')
            return -1, -1, -1

        v1 = self.pretrained_embed(query1_id)
        v2 = self.pretrained_embed(query2_id)
        pre_sim = F.cosine_similarity(v1, v2, dim=0).item()

        v1 = self.deno_space.decomposed(query1_id)
        v2 = self.deno_space.decomposed(query2_id)
        deno_sim = F.cosine_similarity(v1, v2, dim=0).item()

        v1 = self.cono_space.decomposed(query1_id)
        v2 = self.cono_space.decomposed(query2_id)
        cono_sim = F.cosine_similarity(v1, v2, dim=0).item()

        return pre_sim, deno_sim, cono_sim

    def export_embeddings(self, out_path: Path) -> Tuple[Matrix, Matrix]:
        raise NotImplementedError


class LabeledSentences(torch.utils.data.Dataset):

    def __init__(self, config: 'IdealGroundedConfig'):
        super().__init__()
        with open(config.corpus_path, 'rb') as corpus_file:
            preprocessed = pickle.load(corpus_file)
        self.word_to_id = preprocessed['word_to_id']
        self.id_to_word = preprocessed['id_to_word']
        self.deno_to_id = preprocessed['deno_to_id']
        self.id_to_deno = preprocessed['id_to_deno']
        self.ground: Dict[str, GroundedWord] = preprocessed['ground']

        self.train_seq: List[List[int]] = preprocessed['train_sent_word_ids']
        self.train_deno_labels: List[int] = preprocessed['train_deno_labels']
        self.train_cono_labels: List[int] = preprocessed['train_cono_labels']

        self.dev_seq = rnn.pad_sequence(
            [torch.tensor(seq) for seq in preprocessed['dev_sent_word_ids']],
            batch_first=True)
        self.dev_deno_labels = torch.tensor(preprocessed['dev_deno_labels'])
        self.dev_cono_labels = torch.tensor(preprocessed['dev_cono_labels'])

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

    def __len__(self) -> int:
        return len(self.train_seq)

    def __getitem__(self, index: int) -> Tuple[List[int], int, int]:
        return (
            self.train_seq[index],
            self.train_deno_labels[index],
            self.train_cono_labels[index])

    @staticmethod
    def collate(
            batch: List[Tuple[List[int], int, int]]
            ) -> Tuple[Matrix, Vector, Vector]:
        # seq_word_ids = torch.cat([torch.tensor(w) for w, _, _ in batch])
        seq_word_ids = [torch.tensor(w) for w, _, _ in batch]
        deno_labels = torch.tensor([d for _, d, _ in batch])
        cono_labels = torch.tensor([c for _, _, c in batch])
        return (
            rnn.pad_sequence(seq_word_ids, batch_first=True),
            deno_labels,
            cono_labels)


class IdealGroundedExperiment(Experiment):

    def __init__(self, config: 'IdealGroundedConfig'):
        super().__init__(config)
        self.data = LabeledSentences(config)
        self.dataloader = torch.utils.data.DataLoader(
            self.data,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.data.collate,
            num_workers=config.num_dataloader_threads,
            pin_memory=config.pin_memory)
        self.model = Recomposer(config, self.data)
        model = self.model

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)  # param.data)

        self.DS_deno_optimizer = config.optimizer(
            model.deno_space.deno_probe.parameters(), lr=config.learning_rate)
        self.DS_cono_optimizer = config.optimizer(
            model.deno_space.cono_probe.parameters(), lr=config.learning_rate)

        self.CS_deno_optimizer = config.optimizer(
            model.cono_space.deno_probe.parameters(), lr=config.learning_rate)
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
        dev_Hd, dev_Hc = model.deno_space.homogeneity(self.data.dev_ids)
        test_Hd, test_Hc = model.deno_space.homogeneity(self.data.test_ids)
        rand_Hd, rand_Hc = model.deno_space.homogeneity(self.data.rand_ids)
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
        deno_labels = batch[1].to(self.device)
        cono_labels = batch[2].to(self.device)

        # Update probes with proper (non-adversarial) losses
        model.zero_grad()
        DS_deno_probe, DS_cono_probe, DS_cono_adver, _ = model.deno_space(
            seq_word_ids, deno_labels, cono_labels)
        DS_deno_probe.backward(retain_graph=True)
        DS_cono_probe.backward()
        self.DS_deno_optimizer.step()
        self.DS_cono_optimizer.step()

        model.zero_grad()
        CS_deno_probe, CS_deno_adver, CS_cono_probe, _ = model.cono_space(
            seq_word_ids, deno_labels, cono_labels)
        CS_deno_probe.backward(retain_graph=True)
        CS_cono_probe.backward()
        self.CS_deno_optimizer.step()
        self.CS_cono_optimizer.step()

        model.zero_grad()
        (L_joint, L_R,
            DS_decomp, DS_deno_probe, DS_cono_probe, DS_cono_adver,
            CS_decomp, CS_deno_probe, CS_deno_adver, CS_cono_probe) = model(
            seq_word_ids, deno_labels, cono_labels)
        L_joint.backward()
        self.joint_optimizer.step()
        self.R_optimizer.step()

        if batch_index % self.config.update_tensorboard == 0:
            D_deno_acc, D_cono_acc, C_deno_acc, C_cono_acc = model.accuracy(
                seq_word_ids, deno_labels, cono_labels)
            self.update_tensorboard({
                'Denotation Decomposer/deno_loss': DS_deno_probe,
                'Denotation Decomposer/cono_loss_proper': DS_cono_probe,
                'Denotation Decomposer/cono_loss_adversary': DS_cono_adver,
                'Denotation Decomposer/combined loss': DS_decomp,
                'Denotation Decomposer/accuracy_train_deno': D_deno_acc,
                'Denotation Decomposer/accuracy_train_cono': D_cono_acc,

                'Connotation Decomposer/cono_loss': CS_cono_probe,
                'Connotation Decomposer/deno_loss_proper': CS_deno_probe,
                'Connotation Decomposer/deno_loss_adversary': CS_deno_adver,
                'Connotation Decomposer/combined_loss': CS_decomp,
                'Connotation Decomposer/accuracy_train_deno': C_deno_acc,
                'Connotation Decomposer/accuracy_train_cono': C_cono_acc,

                'Joint/Loss': L_joint,
                'Joint/Recomposer': L_R
            })

        if batch_index % self.config.eval_dev_set == 0:
            D_deno_acc, D_cono_acc, C_deno_acc, C_cono_acc = model.accuracy(
                self.data.dev_seq.to(self.device),
                self.data.dev_deno_labels.to(self.device),
                self.data.dev_cono_labels.to(self.device))
            self.update_tensorboard({
                'Denotation Decomposer/accuracy_dev_deno': D_deno_acc,
                'Denotation Decomposer/accuracy_dev_cono': D_cono_acc,
                'Connotation Decomposer/accuracy_dev_deno': C_deno_acc,
                'Connotation Decomposer/accuracy_dev_cono': C_cono_acc})
            # self.update_tensorboard(
            #     model.tabulate(self.data.dev_ids, prefix='Dev Homogeneity/'))
            # self.update_tensorboard(
            #     model.tabulate(self.data.test_ids, prefix='Random Homogeneity/'))
            # self.update_tensorboard(
            #     model.tabulate(self.data.test_ids, prefix='Test Homogeneity/'))

    def eval_step(self, epoch_index: int) -> None:
        PE = self.model.PE_homogeneity
        DS_Hd, DS_Hc, CS_Hd, CS_Hc = self.model.homogeneity(self.data.dev_ids)
        self.update_tensorboard({
            'Homogeneity Diff Dev/DS Hdeno': DS_Hd - PE['dev Hd'],
            'Homogeneity Diff Dev/DS Hcono': DS_Hc - PE['dev Hc'],
            'Homogeneity Diff Dev/CS Hdeno': CS_Hd - PE['dev Hd'],
            'Homogeneity Diff Dev/CS Hcono': CS_Hc - PE['dev Hc'],
            }, manual_step=epoch_index)
        DS_Hd, DS_Hc, CS_Hd, CS_Hc = self.model.homogeneity(self.data.test_ids)
        self.update_tensorboard({
            'Homogeneity Diff Test/DS Hdeno': DS_Hd - PE['test Hd'],
            'Homogeneity Diff Test/DS Hcono': DS_Hc - PE['test Hc'],
            'Homogeneity Diff Test/CS Hdeno': CS_Hd - PE['test Hd'],
            'Homogeneity Diff Test/CS Hcono': CS_Hc - PE['test Hc'],
            }, manual_step=epoch_index)
        DS_Hd, DS_Hc, CS_Hd, CS_Hc = self.model.homogeneity(self.data.rand_ids)
        self.update_tensorboard({
            'Homogeneity Diff Random/DS Hdeno': DS_Hd - PE['rand Hd'],
            'Homogeneity Diff Random/DS Hcono': DS_Hc - PE['rand Hc'],
            'Homogeneity Diff Random/CS Hdeno': CS_Hd - PE['rand Hd'],
            'Homogeneity Diff Random/CS Hcono': CS_Hc - PE['rand Hc'],
            }, manual_step=epoch_index)

    def train(self) -> None:
        config = self.config
        # # For debugging only
        # self.save_everything(self.config.output_dir / 'init_recomposer.pt')
        # raise SystemExit

        if config.print_stats:
            epoch_pbar = tqdm(
                range(1, config.num_epochs + 1),
                desc='Epochs')
        else:
            epoch_pbar = tqdm(
                range(1, config.num_epochs + 1),
                desc=config.output_dir.name)

        for epoch_index in epoch_pbar:
            if config.print_stats:
                batches = tqdm(
                    enumerate(self.dataloader),
                    total=len(self.dataloader),
                    mininterval=config.progress_bar_refresh_rate,
                    desc='Batches')
            else:
                batches = enumerate(self.dataloader)

            for batch_index, batch in batches:
                self.train_step(batch_index, batch)
                self.tb_global_step += 1
            self.auto_save(epoch_index)

            self.eval_step(epoch_index)

            if config.print_stats:
                self.print_timestamp(epoch_index)
            # if config.export_error_analysis:
            #     if (epoch_index % config.export_error_analysis == 0
            #             or epoch_index == 1):
            #         # model.all_vocab_connotation(os.path.join(
            #         #     config.output_dir, f'vocab_cono_epoch{epoch_index}.txt'))
            #         analysis_path = os.path.join(
            #             config.output_dir, f'error_analysis_epoch{epoch_index}.tsv')
            #         deno_accuracy, cono_accuracy = model.accuracy(
            #             self.data.dev_seq.to(self.device),
            #             self.data.dev_deno_labels.to(self.device),
            #             self.data.dev_cono_labels.to(self.device),
            #             error_analysis_path=analysis_path)


@dataclass
class IdealGroundedConfig():
    corpus_path: Path = Path('../../data/ready/CR_topic_context3/train_data.pickle')
    rand_path: Path = Path('../../data/ready/CR_topic_context3/eval_words_random.txt')
    dev_path: Path = Path('../../data/ready/CR_topic_context3/0.7partisan_dev_words.txt')
    test_path: Path = Path('../../data/ready/CR_topic_context3/0.7partisan_test_words.txt')
    num_deno_classes: int = 41
    num_cono_classes: int = 2

    # corpus_path: str = '../../data/ready/CR_bill_context3/train_data.pickle'
    # rand_path: Path = Path('../../data/ready/CR_bill_context3/eval_words_random.txt')
    # dev_path: Path = Path('../../data/ready/CR_bill_context3/0.7partisan_dev_words.txt')
    # test_path: Path = Path('../../data/ready/CR_bill_context3/0.7partisan_test_words.txt')
    # num_deno_classes: int = 1029
    # num_cono_classes: int = 2

    pretrained_embed_path: Optional[Path] = Path(
        '../../data/pretrained_word2vec/CR_bill_topic_context3.txt')

    output_dir: Path = Path('../../results/debug')
    device: torch.device = torch.device('cuda')
    # debug_subset_corpus: Optional[int] = None

    num_dataloader_threads: int = 0
    pin_memory: bool = True

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
    batch_size: int = 1024
    embed_size: int = 300
    num_epochs: int = 150

    optimizer: torch.optim.Optimizer = torch.optim.Adam
    learning_rate: float = 1e-4
    # clip_grad_norm: float = 10.0

    # Housekeeping
    # export_error_analysis: Optional[int] = 1  # per epoch
    update_tensorboard: int = 1000  # per batch
    print_stats: Optional[int] = 10_000  # per batch
    eval_dev_set: int = 100_000  # per batch
    progress_bar_refresh_rate: int = 1  # per second
    clear_tensorboard_log_in_output_dir: bool = True
    delete_all_exisiting_files_in_output_dir: bool = False
    auto_save_per_epoch: Optional[int] = 10
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
        parser.add_argument(
            '-lr', '--learning-rate', action='store', type=float)
        parser.add_argument(
            '-bs', '--batch-size', action='store', type=int)
        parser.add_argument(
            '-ep', '--num-epochs', action='store', type=int)
        parser.add_argument(
            '-pe', '--pretrained-embed-path', action='store', type=Path)
        parser.add_argument(
            '-sv', '--auto-save-per-epoch', action='store', type=int)
        parser.parse_args(namespace=self)

        if self.architecture == 'linear':
            self.deno_probe = nn.Linear(300, self.num_deno_classes)
            self.cono_probe = nn.Linear(300, self.num_cono_classes)
        if self.architecture == 'MLP1':
            self.deno_probe = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, self.num_deno_classes))
            self.cono_probe = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, self.num_cono_classes))
        elif self.architecture == 'MLP2':
            self.deno_probe = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, self.num_deno_classes),
                nn.ReLU())
            self.cono_probe = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, self.num_cono_classes),
                nn.ReLU())
        elif self.architecture == 'MLP2_large':
            self.deno_probe = nn.Sequential(
                nn.Linear(300, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.num_deno_classes),
                nn.ReLU())
            self.cono_probe = nn.Sequential(
                nn.Linear(300, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.num_cono_classes),
                nn.ReLU())
        elif self.architecture == 'MLP3':
            self.deno_probe = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, self.num_deno_classes))
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
        elif self.architecture == 'MLP4':
            self.deno_probe = nn.Sequential(
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
                nn.Linear(300, self.num_deno_classes))
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
        elif self.architecture == 'MLP4_large':
            self.deno_probe = nn.Sequential(
                nn.Linear(300, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.num_deno_classes))
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
        assert self.deno_probe[-1].out_features == self.num_deno_classes


def main() -> None:
    config = IdealGroundedConfig()
    black_box = IdealGroundedExperiment(config)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
