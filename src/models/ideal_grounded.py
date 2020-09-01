import argparse
import pickle
from statistics import mean
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Counter, Optional, Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn
from tqdm import tqdm
import editdistance

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
            grounding: Dict[str, Dict[str, Any]],
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
        self.grounding = grounding

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
        proper_deno_loss = F.nll_loss(deno_log_prob, deno_labels)

        cono_logits = self.cono_probe(seq_repr)
        cono_log_prob = F.log_softmax(cono_logits, dim=1)
        proper_cono_loss = F.nll_loss(cono_log_prob, cono_labels)

        if self.preserve == 'deno':  # DS removing connotation (gamma < 0)
            uniform_dist = torch.full_like(cono_log_prob, 1 / self.num_cono_classes)
            adversary_cono_loss = F.kl_div(cono_log_prob, uniform_dist, reduction='batchmean')
            decomposer_loss = torch.sigmoid(proper_deno_loss) + torch.sigmoid(adversary_cono_loss)
            adversary_deno_loss = 0  # placeholder
        else:  # CS removing denotation
            uniform_dist = torch.full_like(deno_log_prob, 1 / self.num_deno_classes)
            adversary_deno_loss = F.kl_div(deno_log_prob, uniform_dist, reduction='batchmean')
            decomposer_loss = torch.sigmoid(adversary_deno_loss) + torch.sigmoid(proper_cono_loss)
            adversary_cono_loss = 0

        return (decomposer_loss,
                proper_deno_loss, adversary_deno_loss,
                proper_cono_loss, adversary_cono_loss, seq_word_vecs)

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

    def ground(
            self,
            word: str
            ) -> Tuple[str, int]:  # TODO
        deno_label = self.grounding[word]['majority_deno']
        cono_continuous = self.grounding[word]['R_ratio']
        if cono_continuous < 0.5:
            cono_discrete = 0
        else:
            cono_discrete = 1
        # if cono_continuous < 0.2:
        #     cono_discrete = 0
        # elif cono_continuous < 0.8:
        #     cono_discrete = 1
        # else:
        #     cono_discrete = 2
        return deno_label, cono_discrete

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

            query_deno, query_cono = self.ground(query_word)
            same_deno = 0
            same_cono = 0
            for nid in neighbor_ids:
                neighbor_word = self.id_to_word[nid]
                neighbor_deno, neighbor_cono = self.ground(neighbor_word)
                if neighbor_deno == query_deno:
                    same_deno += 1
                if neighbor_cono == query_cono:
                    same_cono += 1
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
            grounding=data.grounding,
            device=self.device)

        self.cono_space = Decomposer(
            preserve='cono',
            initial_space=self.pretrained_embed.weight,
            deno_probe=config.deno_probe,
            cono_probe=config.cono_probe,
            id_to_word=data.id_to_word,
            grounding=data.grounding,
            device=self.device)

        # Recomposer
        # self.recomposer = nn.Linear(600, 300)
        self.rho = config.recomposer_rho
        self.to(self.device)

        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.grounding = data.grounding

    def forward(
            self,
            seq_word_ids: Matrix,
            deno_labels: Vector,
            cono_labels: Vector
            ) -> Tuple[Scalar, ...]:
        L_DS, DS_dp, DS_da, DS_cp, DS_ca, deno_vecs = self.deno_space(
            seq_word_ids, deno_labels, cono_labels)
        L_CS, CS_dp, CS_da, CS_cp, CS_ca, cono_vecs = self.cono_space(
            seq_word_ids, deno_labels, cono_labels)

        # recomposed = self.recomposer(torch.cat((deno_vecs, cono_vecs), dim=-1))
        recomposed = deno_vecs + cono_vecs  # cosine similarity ignores magnitude
        pretrained = self.pretrained_embed(seq_word_ids)
        L_R = 1 - F.cosine_similarity(recomposed, pretrained, dim=-1).mean()

        L_joint = L_DS + L_CS + self.rho * L_R
        return L_joint, L_R, DS_dp, DS_cp, DS_ca, L_CS, CS_dp, CS_da, CS_cp

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
            query_ids: Vector,
            prefix: str = '',
            suffix: str = '',
            rounding: int = 4,
            top_k: int = 10
            ) -> Dict[str, float]:
        row = {}
        DS_Hd, DS_Hc, CS_Hd, CS_Hc = self.homogeneity(query_ids, top_k=top_k)
        row['DS Hdeno'] = DS_Hd
        row['DS Hcono'] = DS_Hc
        row['CS Hdeno'] = CS_Hd
        row['CS Hcono'] = CS_Hc
        # row['IntraDS Hd - Hc'] = DS_Hd - DS_Hc
        # row['IntraCS Hc - Hd'] = CS_Hc - CS_Hd
        # row['mean IntraS quality'] = (row['IntraDS Hd - Hc'] + row['IntraCS Hc - Hd']) / 2

        # row['main diagnoal trace'] = (DS_Hd + CS_Hc) / 2  # max all preservation
        # row['nondiagnoal entries negative sum'] = (-DS_Hc - CS_Hd) / 2  # min all discarded
        # row['flattened weighted sum'] = row['main diagnoal trace'] + row['nondiagnoal entries negative sum']

        # row['Inter DS Hd - CS Hd'] = DS_Hd - CS_Hd
        # row['Inter CS Hc - DS Hc'] = CS_Hc - DS_Hc
        # row['mean InterS quality'] = (row['Inter DS Hd - CS Hd'] + row['Inter CS Hc - DS Hc']) / 2
        return {prefix + key + suffix: round(val, rounding)
                for key, val in row.items()}

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
        # word -> Counter[deno/cono]
        self.grounding: Dict[str, Dict[str, Any]] = preprocessed['grounding']
        # deno/cono -> Counter[word]
        self.counts: Dict[str, Counter[str]] = preprocessed['counts']

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
        self.DS_decomp_optimizer = config.optimizer(
            model.deno_space.decomposed.parameters(),
            lr=config.learning_rate)
        self.DS_deno_optimizer = config.optimizer(
            model.deno_space.deno_probe.parameters(),
            lr=config.learning_rate)
        self.DS_cono_optimizer = config.optimizer(
            model.deno_space.cono_probe.parameters(),
            lr=config.learning_rate)

        self.CS_decomp_optimizer = config.optimizer(
            model.cono_space.decomposed.parameters(),
            lr=config.learning_rate)
        self.CS_deno_optimizer = config.optimizer(
            model.cono_space.deno_probe.parameters(),
            lr=config.learning_rate)
        self.CS_cono_optimizer = config.optimizer(
            model.cono_space.cono_probe.parameters(),
            lr=config.learning_rate)
        # self.R_optimizer = config.optimizer(
        #     model.recomposer.parameters(),
        #     lr=config.learning_rate)

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

        model.zero_grad()
        L_joint, L_R, DS_dp, DS_cp, DS_ca, L_CS, CS_dp, CS_da, CS_cp = model(
            seq_word_ids, deno_labels, cono_labels)
        L_joint.backward()
        self.DS_decomp_optimizer.step()
        self.CS_decomp_optimizer.step()

        # Update probes with proper losses
        # TODO test reordering
        model.zero_grad()
        L_DS, DS_dp, DS_da, DS_cp, DS_ca, _ = model.deno_space(
            seq_word_ids, deno_labels, cono_labels)
        DS_dp.backward(retain_graph=True)
        self.DS_deno_optimizer.step()
        # model.zero_grad()
        DS_cp.backward()
        self.DS_cono_optimizer.step()

        model.zero_grad()
        L_CS, CS_dp, CS_da, CS_cp, CS_ca, _ = model.cono_space(
            seq_word_ids, deno_labels, cono_labels)
        CS_dp.backward(retain_graph=True)
        self.CS_deno_optimizer.step()
        # model.zero_grad()
        CS_cp.backward()
        self.CS_cono_optimizer.step()

        if batch_index % self.config.update_tensorboard == 0:
            D_deno_acc, D_cono_acc, C_deno_acc, C_cono_acc = model.accuracy(
                seq_word_ids, deno_labels, cono_labels)
            self.update_tensorboard({
                'Denotation Decomposer/deno_loss': DS_dp,
                'Denotation Decomposer/cono_loss_proper': DS_cp,
                'Denotation Decomposer/cono_loss_adversary': DS_ca,
                'Denotation Decomposer/combined loss': L_DS,
                'Denotation Decomposer/accuracy_train_deno': D_deno_acc,
                'Denotation Decomposer/accuracy_train_cono': D_cono_acc,

                'Connotation Decomposer/cono_loss': CS_cp,
                'Connotation Decomposer/deno_loss_proper': CS_dp,
                'Connotation Decomposer/deno_loss_adversary': CS_da,
                'Connotation Decomposer/combined_loss': L_CS,
                'Connotation Decomposer/accuracy_train_deno': C_deno_acc,
                'Connotation Decomposer/accuracy_train_cono': C_cono_acc,

                'Joint/Loss': L_joint,
                'Joint/Recomposer': L_R
            })

        if batch_index % self.config.eval_dev_set == 0:
            self.eval_step()
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

    def eval_step(self) -> None:
        PE = self.model.PE_homogeneity
        DS_Hd, DS_Hc, CS_Hd, CS_Hc = self.model.homogeneity(self.data.dev_ids)
        self.update_tensorboard({
            'Homogeneity Diff Dev/DS Hdeno': DS_Hd - PE['dev Hd'],
            'Homogeneity Diff Dev/DS Hcono': DS_Hc - PE['dev Hc'],
            'Homogeneity Diff Dev/CS Hdeno': CS_Hd - PE['dev Hd'],
            'Homogeneity Diff Dev/CS Hcono': CS_Hc - PE['dev Hc'],
        })
        DS_Hd, DS_Hc, CS_Hd, CS_Hc = self.model.homogeneity(self.data.test_ids)
        self.update_tensorboard({
            'Homogeneity Diff Test/DS Hdeno': DS_Hd - PE['test Hd'],
            'Homogeneity Diff Test/DS Hcono': DS_Hc - PE['test Hc'],
            'Homogeneity Diff Test/CS Hdeno': CS_Hd - PE['test Hd'],
            'Homogeneity Diff Test/CS Hcono': CS_Hc - PE['test Hc'],
        })
        DS_Hd, DS_Hc, CS_Hd, CS_Hc = self.model.homogeneity(self.data.rand_ids)
        self.update_tensorboard({
            'Homogeneity Diff Random/DS Hdeno': DS_Hd - PE['rand Hd'],
            'Homogeneity Diff Random/DS Hcono': DS_Hc - PE['rand Hc'],
            'Homogeneity Diff Random/CS Hdeno': CS_Hd - PE['rand Hd'],
            'Homogeneity Diff Random/CS Hcono': CS_Hc - PE['rand Hc'],
        })

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
    corpus_path: Path = Path('../../data/processed/bill_mentions/topic_deno/train_data.pickle')
    num_deno_classes: int = 41
    num_cono_classes: int = 2

    # input_dir: str = '../../data/processed/bill_mentions/title_deno_context3'
    # num_deno_classes: int = 1029

    # input_dir: str = '../../data/processed/bill_mentions/title_deno_context5'
    # num_deno_classes: int = 1027

    # input_dir: str = '../../data/processed/bill_mentions/title_deno'

    rand_path: Path = Path('../../data/ellie/rand_sample.cr.txt')
    dev_path: Path = Path('../../data/ellie/partisan_sample_val.cr.txt')
    test_path: Path = Path('../../data/ellie/partisan_sample.cr.txt')
    pretrained_embed_path: Optional[Path] = Path('../../data/pretrained_word2vec/CR_ctx3_HS.txt')

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

    architecture: str = 'L4R'
    batch_size: int = 512
    embed_size: int = 300
    num_epochs: int = 100
    # encoder_update_cycle: int = 1  # per batch
    # decoder_update_cycle: int = 1  # per batch

    optimizer: torch.optim.Optimizer = torch.optim.Adam
    learning_rate: float = 1e-3
    clip_grad_norm: float = 10.0

    # Housekeeping
    # export_error_analysis: Optional[int] = 1  # per epoch
    update_tensorboard: int = 1000  # per batch
    print_stats: Optional[int] = 10_000  # per batch
    eval_dev_set: int = 100_000  # per batch
    progress_bar_refresh_rate: int = 5  # per second
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
        # parser.add_argument(
        #     '-dd', '--deno-delta', action='store', type=float)
        # parser.add_argument(
        #     '-dg', '--deno-gamma', action='store', type=float)
        # parser.add_argument(
        #     '-cd', '--cono-delta', action='store', type=float)
        # parser.add_argument(
        #     '-cg', '--cono-gamma', action='store', type=float)

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

        if self.architecture == 'L1R':
            self.deno_probe = nn.Sequential(
                nn.Linear(300, self.num_deno_classes),
                nn.ReLU())
            self.cono_probe = nn.Sequential(
                nn.Linear(300, self.num_cono_classes),
                nn.ReLU())
        elif self.architecture == 'L2R':
            self.deno_probe = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, self.num_deno_classes),
                nn.ReLU())
            self.cono_probe = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, self.num_cono_classes),
                nn.ReLU())
        elif self.architecture == 'L3R':
            self.deno_probe = nn.Sequential(
                nn.Linear(300, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, self.num_deno_classes),
                nn.ReLU())
            self.cono_probe = nn.Sequential(
                nn.Linear(300, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, self.num_cono_classes),
                nn.ReLU())
        elif self.architecture == 'L4':
            self.deno_probe = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.AlphaDropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.SELU(),
                nn.AlphaDropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, self.num_deno_classes),
                nn.SELU())
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
        elif self.architecture == 'L4RL':
            self.deno_probe = nn.Sequential(
                nn.Linear(300, 1024),
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
