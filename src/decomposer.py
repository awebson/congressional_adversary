import argparse
import random
import pickle
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Counter, Optional

from ortho_basis_th import OrthoBasis

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn
from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import homogeneity_score
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

new_base_path = "/data/people/tberckma/congressional_adversary/congressional_adversary/"
new_base_path = "/Users/tberckma/Research/newcong_ad_data/"
#new_base_path = "/data/people/tberckma/new_congad_data/"

class Decomposer(nn.Module):

    def __init__(
            self,
            config: 'DecomposerConfig',
            data: 'LabeledSentences'):
        super().__init__()
        self.init_embedding(config, data.word_to_id)
        self.num_deno_classes = config.num_deno_classes
        self.num_cono_classes = config.num_cono_classes
        assert self.num_deno_classes == len(data.deno_to_id)

        # Dennotation Loss: Skip-Gram Negative Sampling
        self.deno_decoder = config.deno_architecture
        # assert self.deno_decoder[0].in_features == repr_size
        # assert self.deno_decoder[-2].out_features == self.num_deno_classes

        # Connotation Loss: Party Classifier
        self.cono_decoder = config.cono_architecture
        
        # Reverse classifiers
        self.deno_rev_decoder = config.deno_rev_architecture 
        self.cono_rev_decoder = config.cono_rev_architecture
        # assert self.cono_decoder[-2].out_features == self.num_cono_classes

        self.ortho_basis = OrthoBasis(config.embed_size, config.device)

        self.delta = config.delta
        self.gamma = config.gamma
        self.device = config.device
        self.to(self.device)

        self.deno_to_id = data.deno_to_id  # for homogeneity evaluation
        self.id_to_deno = data.id_to_deno  # for error analysis
        # self.graph_labels = [
        #     data.id_to_deno[i]
        #     for i in range(len(data.id_to_deno))]

        # Initailize neighbor cono homogeneity eval partisan vocabulary
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.counts = data.counts  # saved just in case
        self.grounding = data.grounding

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

    def forward(
            self,
            seq_word_ids: Matrix,
            deno_labels: Vector,
            cono_labels: Vector
            ) -> Scalar:
        seq_vecs: R3Tensor = self.embedding(seq_word_ids)
        seq_repr: Matrix = torch.mean(seq_vecs, dim=1)

        c_vecs, d_vecs = self.ortho_basis(seq_repr)

        deno_logits = self.deno_decoder(d_vecs)
        deno_log_prob = F.log_softmax(deno_logits, dim=1)
        proper_deno_loss = F.nll_loss(deno_log_prob, deno_labels)

        cono_logits = self.cono_decoder(c_vecs)
        cono_log_prob = F.log_softmax(cono_logits, dim=1)
        proper_cono_loss = F.nll_loss(cono_log_prob, cono_labels)
        
        joint_loss = proper_deno_loss + proper_cono_loss

        # Reverse classifiers
        deno_rev_logits =  self.deno_rev_decoder(c_vecs.detach().clone())
        deno_rev_log_prob = F.log_softmax(deno_rev_logits, dim=1)
        proper_deno_rev_loss = F.nll_loss(deno_rev_log_prob, deno_labels)
        cono_rev_logits = self.cono_rev_decoder(d_vecs.detach().clone())
        cono_rev_log_prob = F.log_softmax(cono_rev_logits, dim=1)
        proper_cono_rev_loss = F.nll_loss(cono_rev_log_prob, cono_labels)

        #if self.gamma < 0:  # DS removing connotation
        #    uniform_dist = torch.full_like(cono_log_prob, 1 / self.num_cono_classes)
        #    adversary_cono_loss = F.kl_div(cono_log_prob, uniform_dist, reduction='batchmean')
        #    decomposer_loss = torch.sigmoid(proper_deno_loss) + torch.sigmoid(adversary_cono_loss)
        #    adversary_deno_loss = 0  # placeholder
        #else:  # CS removing denotation
        #    uniform_dist = torch.full_like(deno_log_prob, 1 / self.num_deno_classes)
        #    adversary_deno_loss = F.kl_div(deno_log_prob, uniform_dist, reduction='batchmean')
        #    decomposer_loss = torch.sigmoid(adversary_deno_loss) + torch.sigmoid(proper_cono_loss)
        #    adversary_cono_loss = 0

        return joint_loss, proper_deno_loss, proper_cono_loss, proper_deno_rev_loss, proper_cono_rev_loss

        #if recompose:
        #    return decomposer_loss, proper_deno_loss, adversary_deno_loss, proper_cono_loss, adversary_cono_loss, seq_vecs
        #else:
        #    return decomposer_loss, proper_deno_loss, adversary_deno_loss, proper_cono_loss, adversary_cono_loss

    def predict(self, seq_word_ids: Vector) -> Vector:
        self.eval()
        with torch.no_grad():
            word_vecs: R3Tensor = self.embedding(seq_word_ids)
            seq_repr: Matrix = torch.mean(word_vecs, dim=1)
            
            c_vecs, d_vecs = self.ortho_basis(seq_repr)

            deno = self.deno_decoder(d_vecs)
            cono = self.cono_decoder(c_vecs)

            deno_rev = self.deno_rev_decoder(c_vecs)
            cono_rev = self.cono_rev_decoder(d_vecs)

            deno_conf = nn.functional.softmax(deno, dim=1)
            cono_conf = nn.functional.softmax(cono, dim=1)

            deno_rev_conf = nn.functional.softmax(deno_rev, dim=1)
            cono_rev_conf = nn.functional.softmax(cono_rev, dim=1)
        self.train()
        return deno_conf, cono_conf, deno_rev_conf, cono_rev_conf

    def accuracy(
            self,
            seq_word_ids: Matrix,
            deno_labels: Vector,
            cono_labels: Vector,
            error_analysis_path: Optional[str] = None
            ) -> Tuple[float, float]:
        deno_conf, cono_conf, deno_rev_conf, cono_rev_conf = self.predict(seq_word_ids)
        deno_predictions = deno_conf.argmax(dim=1)
        cono_predictions = cono_conf.argmax(dim=1)
        deno_rev_predictions = deno_rev_conf.argmax(dim=1)
        cono_rev_predictions = cono_rev_conf.argmax(dim=1)
        # # Random Guess Baseline
        # deno_predictions = torch.randint_like(deno_labels, high=len(self.deno_to_id))
        # # Majority Class Baseline
        # majority_label = self.deno_to_id['Health']
        # deno_predictions = torch.full_like(deno_labels, majority_label)

        deno_correct_indicies = deno_predictions.eq(deno_labels)
        cono_correct_indicies = cono_predictions.eq(cono_labels)
        deno_rev_correct_indicies = deno_rev_predictions.eq(deno_labels)
        cono_rev_correct_indicies = cono_rev_predictions.eq(cono_labels)
        deno_accuracy = deno_correct_indicies.float().mean().item()
        cono_accuracy = cono_correct_indicies.float().mean().item()
        deno_rev_accuracy = deno_rev_correct_indicies.float().mean().item()
        cono_rev_accuracy = cono_rev_correct_indicies.float().mean().item()

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
        return deno_accuracy, cono_accuracy, deno_rev_accuracy, cono_rev_accuracy

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


    @staticmethod
    def discretize_cono(skew: float) -> int:
        if skew < 0.5:
            return 0
        else:
            return 1

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
        top_neighbor_ids = self.nearest_neighbors(query_ids, top_k + 5)
        ground = self.grounding
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

            query_deno = ground[query_word]['majority_deno']
            query_cono = self.discretize_cono(ground[query_word]['R_ratio'])
            same_deno = 0
            same_cono = 0
            for nid in neighbor_ids:
                neighbor_word = self.id_to_word[nid]
                if ground[neighbor_word]['majority_deno'] == query_deno:
                    same_deno += 1
                if self.discretize_cono(ground[neighbor_word]['R_ratio']) == query_cono:
                    same_cono += 1
            deno_homogeneity.append(same_deno / len(neighbor_ids))
            cono_homogeneity.append(same_cono / len(neighbor_ids))
        return np.mean(deno_homogeneity), np.mean(cono_homogeneity)

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


class LabeledSentences(Dataset):

    def __init__(self, config: 'DecomposerConfig'):
        corpus_path = os.path.join(config.input_dir, 'train_data.pickle')
        with open(corpus_path, 'rb') as corpus_file:
            preprocessed = pickle.load(corpus_file)
        self.word_to_id = preprocessed['word_to_id']
        self.id_to_word = preprocessed['id_to_word']
        self.deno_to_id = preprocessed['deno_to_id']
        self.id_to_deno = preprocessed['id_to_deno']
        # word -> Counter[deno/cono]
        self.grounding: Dict[str, Counter[str]] = preprocessed['grounding']
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

        self.deno_optimizer = config.optimizer(
            self.model.deno_decoder.parameters(),
            lr=config.learning_rate)
        self.cono_optimizer = config.optimizer(
            self.model.cono_decoder.parameters(),
            lr=config.learning_rate)
        #self.decomposer_optimizer = config.optimizer(
        #    self.model.embedding.parameters(),
        #    lr=config.learning_rate)
        self.ortho_optimizer = config.optimizer(
            self.model.ortho_basis.parameters(),
            lr=config.learning_rate)
        
        # Reverse classifiers
        self.deno_rev_optimizer = config.optimizer(
            self.model.deno_rev_decoder.parameters(),
            lr=config.learning_rate)
        self.cono_rev_optimizer = config.optimizer(
            self.model.cono_rev_decoder.parameters(),
            lr=config.learning_rate)

        dev_path = Path(new_base_path + 'data/ellie/partisan_sample_val.cr.txt')
        with open(dev_path) as file:
            self.dev_ids = torch.tensor(
                [self.model.word_to_id[word.strip()] for word in file],
                device=config.device)

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
                seq_word_ids = batch[0].to(self.device)
                deno_labels = batch[1].to(self.device)
                cono_labels = batch[2].to(self.device)

                model.zero_grad()
                L_j, l_dp, l_da, ld_rev, lc_rev = self.model(
                    seq_word_ids, deno_labels, cono_labels)
                # TODO update backprop here
                L_j.backward()
                ld_rev.backward()
                lc_rev.backward()
                
                nn.utils.clip_grad_norm_(model.deno_decoder.parameters(), grad_clip)
                self.deno_optimizer.step()

                nn.utils.clip_grad_norm_(model.cono_decoder.parameters(), grad_clip)
                self.cono_optimizer.step()

                #nn.utils.clip_grad_norm_(model.embedding.parameters(), grad_clip)
                #self.decomposer_optimizer.step()

                nn.utils.clip_grad_norm_(model.ortho_basis.parameters(), grad_clip)
                self.ortho_optimizer.step()

                # Reverse classifiers
                nn.utils.clip_grad_norm_(model.deno_rev_decoder.parameters(), grad_clip)
                self.deno_rev_optimizer.step()
                nn.utils.clip_grad_norm_(model.cono_rev_decoder.parameters(), grad_clip)
                self.cono_rev_optimizer.step()

                if batch_index % config.update_tensorboard == 0:
                    deno_accuracy, cono_accuracy, deno_rev_accuracy, cono_rev_accuracy = self.model.accuracy(
                        seq_word_ids, deno_labels, cono_labels)
                    stats = {
                        'Decomposer/l_dp': l_dp,
                        'Decomposer/l_da': l_da,
                        #'Decomposer/l_dp': l_cp,
                        #'Decomposer/l_da': l_ca,
                        #'Decomposer/overcorrect_loss': l_overcorrect,
                        'Decomposer/accuracy_train_deno': deno_accuracy,
                        'Decomposer/accuracy_train_cono': cono_accuracy,
                        'Decomposer/accuracy_train_deno_rev': deno_rev_accuracy,
                        'Decomposer/accuracy_train_cono_rev': cono_rev_accuracy,
                        'Decomposer/joint_loss': L_j
                    }
                    self.update_tensorboard(stats)
                # if config.print_stats and batch_index % config.print_stats == 0:
                #     self.print_stats(epoch_index, batch_index, stats)
                if batch_index % config.eval_dev_set == 0:
                    self.validation()
                self.tb_global_step += 1
            # End Batches
            # self.lr_scheduler.step()
            self.print_timestamp(epoch_index)
            self.auto_save(epoch_index)

            if config.export_error_analysis:
                if (epoch_index % config.export_error_analysis == 0
                        or epoch_index == 1):
                    # self.model.all_vocab_connotation(os.path.join(
                    #     config.output_dir, f'vocab_cono_epoch{epoch_index}.txt'))
                    analysis_path = os.path.join(
                        config.output_dir, f'error_analysis_epoch{epoch_index}.tsv')
                    deno_accuracy, cono_accuracy, deno_rev_accuracy, cono_dev_accuracy = self.model.accuracy(
                        self.data.dev_seq.to(self.device),
                        self.data.dev_deno_labels.to(self.device),
                        self.data.dev_cono_labels.to(self.device),
                        error_analysis_path=analysis_path)
        # End Epochs

    def validation(self) -> None:
        deno_accuracy, cono_accuracy, deno_rev_accuracy, cono_rev_accuracy = self.model.accuracy(
            self.data.dev_seq.to(self.device),
            self.data.dev_deno_labels.to(self.device),
            self.data.dev_cono_labels.to(self.device))
        #Hdeno, Hcono = self.model.homemade_homogeneity(self.dev_ids)
        self.update_tensorboard({
            # 'Denotation Decomposer/nonpolitical_word_sim_cf_pretrained': deno_check,
            'Decomposer/accuracy_dev_deno': deno_accuracy,
            # 'Connotation Decomposer/nonpolitical_word_sim_cf_pretrained': cono_check,
            'Decomposer/accuracy_dev_cono': cono_accuracy,
            'Decomposer/accuracy_dev_deno_rev': deno_rev_accuracy,
            'Decomposer/accuracy_dev_cono_rev': cono_rev_accuracy,
            'Decomposer/Topic Homogeneity': 0, # Hdeno,
            'Decomposer/Party Homogeneity': 0  #Hcono,
        })


@dataclass
class DecomposerConfig():
    # Essential
    input_dir: Path = Path(new_base_path + 'data/processed/bill_mentions')
    num_deno_classes: int = 41
    num_cono_classes: int = 2

    # input_dir: str = '../data/processed/bill_mentions/title_deno_context3'
    # num_deno_classes: int = 1029

    #input_dir: str = '../data/processed/bill_mentions/title_deno_context5'
    #num_deno_classes: int = 1027

    output_dir: Path = Path('../results/debug')
    #device: torch.device = torch.device('cuda')
    device: torch.device = torch.device('cpu')
    debug_subset_corpus: Optional[int] = None
    # dev_holdout: int = 5_000
    # test_holdout: int = 10_000
    num_dataloader_threads: int = 0
    pin_memory: bool = True

    decomposed_size: int = 300
    delta: float = 1  # denotation classifier weight 𝛿
    gamma: float = 1  # connotation classifier weight 𝛾

    architecture: str = 'L4'
    dropout_p: float = 0
    batch_size: int = 128
    embed_size: int = 300
    num_epochs: int = 50
    encoder_update_cycle: int = 1  # per batch
    decoder_update_cycle: int = 1  # per batch

    pretrained_embedding: Optional[Path] = Path(new_base_path + 'data/pretrained_word2vec/bill_mentions_HS.txt')
    freeze_embedding: bool = True
    # window_radius: int = 5
    # num_negative_samples: int = 10
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
    auto_save_per_epoch: Optional[int] = 5
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
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, self.num_deno_classes),
                nn.SELU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 2),
                nn.SELU())
        elif self.architecture == 'L2':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, self.num_deno_classes),
                nn.SELU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 2),
                nn.SELU())
        elif self.architecture == 'L4':
            def get_l4_architecture(output_width):
                return nn.Sequential(
                    nn.Linear(300, 300),
                    nn.SELU(),
                    nn.AlphaDropout(p=self.dropout_p),
                    nn.Linear(300, 300),
                    nn.SELU(),
                    nn.AlphaDropout(p=self.dropout_p),
                    nn.Linear(300, 300),
                    nn.SELU(),
                    nn.Linear(300, output_width),
                    nn.SELU())
            self.deno_architecture = get_l4_architecture(self.num_deno_classes)
            self.cono_architecture = get_l4_architecture(2)
            
            # Reverse classifiers for orthogonality experiment
            self.deno_rev_architecture = get_l4_architecture(self.num_deno_classes)
            self.cono_rev_architecture = get_l4_architecture(2)
        else:
            raise ValueError('Unknown architecture argument.')


def main() -> None:
    config = DecomposerConfig()
    black_box = DecomposerExperiment(config)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    main()
