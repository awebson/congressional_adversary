import argparse
import random
import sys
from copy import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from decomposer import Decomposer, DecomposerConfig, LabeledSentences, new_base_path
from utils.experiment import Experiment
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor
from gcide import get_full_pos_dict, global_pos_list
from wordcat import word_cat, annotate_heatmap, heatmap

from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

random.seed(42)
torch.manual_seed(42)


class Recomposer(nn.Module):

    def __init__(
            self,
            config: 'RecomposerConfig',
            data: 'LabeledSentences'):
        super().__init__()
        self.delta = config.deno_delta
        self.gamma = config.deno_gamma
        self.device = config.device

        # Denotation Decomposer
        deno_config = copy(config)
        deno_config.delta = config.deno_delta
        deno_config.gamma = config.deno_gamma
        self.deno_decomposer = Decomposer(deno_config, data)

        # Connotation Decomposer
        cono_config = copy(config)
        cono_config.delta = config.cono_delta
        cono_config.gamma = config.cono_gamma
        self.cono_decomposer = Decomposer(cono_config, data)

        # Recomposer
        self.pretrained_embed = self.deno_decomposer.pretrained_embed
        # self.recomposer = nn.Linear(600, 300)
        self.rho = config.recomposer_rho
        self.to(self.device)

        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.grounding = data.grounding

        #dev_path = Path('../data/ellie/partisan_sample_val.cr.txt')
        #with open(dev_path) as file:
        #    self.dev_ids = torch.tensor(
        #        [self.word_to_id[word.strip()] for word in file],
        #        device=self.device)

    def forward(
            self,
            seq_word_ids: Matrix,
            deno_labels: Vector,
            cono_labels: Vector
            ) -> Scalar:
        L_DS, DS_dp, DS_da, DS_cp, DS_ca, deno_vecs = self.deno_decomposer(
            seq_word_ids, deno_labels, cono_labels, recompose=True)
        L_CS, CS_dp, CS_da, CS_cp, CS_ca, cono_vecs = self.cono_decomposer(
            seq_word_ids, deno_labels, cono_labels, recompose=True)

        # recomposed = self.recomposer(torch.cat((deno_vecs, cono_vecs), dim=-1))
        recomposed = deno_vecs + cono_vecs  # cosine similarity ignores magnitude
        pretrained = self.pretrained_embed(seq_word_ids)
        L_R = 1 - nn.functional.cosine_similarity(recomposed, pretrained, dim=-1).mean()

        L_joint = L_DS + L_CS + self.rho * L_R
        return L_joint, L_R, DS_dp, DS_cp, DS_ca, L_CS, CS_dp, CS_da, CS_cp

    def predict(self, seq_word_ids: Vector) -> Tuple[Vector, ...]:
        self.eval()
        D_deno_conf, D_cono_conf = self.deno_decomposer.predict(seq_word_ids)
        C_deno_conf, C_cono_conf = self.cono_decomposer.predict(seq_word_ids)
        self.train()
        return D_deno_conf, D_cono_conf, C_deno_conf, C_cono_conf

    def accuracy(
            self,
            seq_word_ids: Matrix,
            deno_labels: Vector,
            cono_labels: Vector,
            error_analysis_path: Optional[str] = None
            ) -> Tuple[float, ...]:
        D_deno_accuracy, D_cono_accuracy = self.deno_decomposer.accuracy(
            seq_word_ids, deno_labels, cono_labels)
        C_deno_accuracy, C_cono_accuracy = self.cono_decomposer.accuracy(
            seq_word_ids, deno_labels, cono_labels)
        return D_deno_accuracy, D_cono_accuracy, C_deno_accuracy, C_cono_accuracy

    def homemade_homogeneity(
            self,
            query_ids: Vector,
            top_k: int = 10
            ) -> Tuple[float, ...]:
        DS_Hdeno, DS_Hcono = self.deno_decomposer.homemade_homogeneity(query_ids, top_k=top_k)
        CS_Hdeno, CS_Hcono = self.cono_decomposer.homemade_homogeneity(query_ids, top_k=top_k)
        return DS_Hdeno, DS_Hcono, CS_Hdeno, CS_Hcono

    def tabulate(
            self,
            query_ids: Vector,
            suffix: str,
            rounding: int = 4
            ) -> Dict[str, float]:
        row = {}
        D_model = self.deno_decomposer
        C_model = self.cono_decomposer
        DS_Hdeno, DS_Hcono = D_model.homemade_homogeneity(query_ids, top_k=10)
        CS_Hdeno, CS_Hcono = C_model.homemade_homogeneity(query_ids, top_k=10)

        row['DS Hdeno'] = DS_Hdeno
        row['DS Hcono'] = DS_Hcono
        row['CS Hdeno'] = CS_Hdeno
        row['CS Hcono'] = CS_Hcono
        row['IntraDS Hd - Hc'] = DS_Hdeno - DS_Hcono
        row['IntraCS Hc - Hd'] = CS_Hcono - CS_Hdeno
        row['mean IntraS quality'] = (row['IntraDS Hd - Hc'] + row['IntraCS Hc - Hd']) / 2

        row['main diagnoal trace'] = (DS_Hdeno + CS_Hcono) / 2  # max all preservation
        row['nondiagnoal entries negative sum'] = (-DS_Hcono - CS_Hdeno) / 2  # min all discarded
        row['flattened weighted sum'] = row['main diagnoal trace'] + row['nondiagnoal entries negative sum']

        row['Inter DS Hd - CS Hd'] = DS_Hdeno - CS_Hdeno
        row['Inter CS Hc - DS Hc'] = CS_Hcono - DS_Hcono
        row['mean InterS quality'] = (row['Inter DS Hd - CS Hd'] + row['Inter CS Hc - DS Hc']) / 2
        if not suffix:
            return {key: round(val, rounding) for key, val in row.items()}
        else:
            return {key + suffix: round(val, rounding) for key, val in row.items()}



class RecomposerExperiment(Experiment):

    def __init__(self, config: 'RecomposerConfig'):
        super().__init__(config)
        self.data = LabeledSentences(config)
        self.dataloader = DataLoader(
            self.data,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.data.collate,
            num_workers=config.num_dataloader_threads,
            pin_memory=config.pin_memory)
        self.model = Recomposer(config, self.data)

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)  # param.data)
        self.D_decomp_optimizer = config.optimizer(
            self.model.deno_decomposer.embedding.parameters(),
            lr=config.learning_rate)
        self.D_deno_optimizer = config.optimizer(
            self.model.deno_decomposer.deno_decoder.parameters(),
            lr=config.learning_rate)
        self.D_cono_optimizer = config.optimizer(
            self.model.deno_decomposer.cono_decoder.parameters(),
            lr=config.learning_rate)

        self.C_decomp_optimizer = config.optimizer(
            self.model.cono_decomposer.embedding.parameters(),
            lr=config.learning_rate)
        self.C_deno_optimizer = config.optimizer(
            self.model.cono_decomposer.deno_decoder.parameters(),
            lr=config.learning_rate)
        self.C_cono_optimizer = config.optimizer(
            self.model.cono_decomposer.cono_decoder.parameters(),
            lr=config.learning_rate)

        # self.R_optimizer = config.optimizer(
        #     self.model.recomposer.parameters(),
        #     lr=config.learning_rate)

        self.to_be_saved = {
            'config': self.config,
            'model': self.model}

        # from evaluations.helpers import polarized_words
        # self.eval_word_ids = torch.tensor(
        #     [w.word_id for w in polarized_words], device=self.device)

    def train(self) -> None:
        config = self.config
        model = self.model
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
                    total=len(self.dataloader),
                    mininterval=config.progress_bar_refresh_rate,
                    desc='Batches')
            for batch_index, batch in batches:
                seq_word_ids = batch[0].to(self.device)
                deno_labels = batch[1].to(self.device)
                cono_labels = batch[2].to(self.device)

                model.zero_grad()
                L_joint, L_R, DS_dp, DS_cp, DS_ca, L_CS, CS_dp, CS_da, CS_cp = model(
                    seq_word_ids, deno_labels, cono_labels)
                L_joint.backward()
                self.D_decomp_optimizer.step()
                self.C_decomp_optimizer.step()

                model.zero_grad()
                L_DS, DS_dp, DS_da, DS_cp, DS_ca = model.deno_decomposer(
                    seq_word_ids, deno_labels, cono_labels)
                DS_dp.backward(retain_graph=True)
                self.D_deno_optimizer.step()
                # model.zero_grad()
                DS_cp.backward()
                self.D_cono_optimizer.step()

                model.zero_grad()
                L_CS, CS_dp, CS_da, CS_cp, CS_ca = model.cono_decomposer(
                    seq_word_ids, deno_labels, cono_labels)
                CS_dp.backward(retain_graph=True)
                self.C_deno_optimizer.step()
                # model.zero_grad()
                CS_cp.backward()
                self.C_cono_optimizer.step()

                if batch_index % config.update_tensorboard == 0:
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

                if batch_index % config.eval_dev_set == 0:
                    D_deno_acc, D_cono_acc, C_deno_acc, C_cono_acc = model.accuracy(
                        self.data.dev_seq.to(self.device),
                        self.data.dev_deno_labels.to(self.device),
                        self.data.dev_cono_labels.to(self.device))

                    DS_Hdeno, DS_Hcono, CS_Hdeno, CS_Hcono = 0.0, 0.0, 0.0, 0.0 # model.homemade_homogeneity(model.dev_ids)
                    self.update_tensorboard({
                        'Denotation Decomposer/accuracy_dev_deno': D_deno_acc,
                        'Denotation Decomposer/accuracy_dev_cono': D_cono_acc,
                        'Connotation Decomposer/accuracy_dev_deno': C_deno_acc,
                        'Connotation Decomposer/accuracy_dev_cono': C_cono_acc,

                        'Denotation Decomposer/topic homogeneity': DS_Hdeno,
                        'Denotation Decomposer/party homogeneity': DS_Hcono,
                        'Connotation Decomposer/topic homogeneity': CS_Hdeno,
                        'Connotation Decomposer/party homogeneity homemade': CS_Hcono,
                    })
                self.tb_global_step += 1
            # End Batches
            # self.lr_scheduler.step()
            self.print_timestamp(epoch_index)
            self.auto_save(epoch_index)
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
        # End Epochs


@dataclass
class RecomposerConfig():
    # Essential
    # input_dir: Path = Path('../data/processed/bill_mentions/topic_deno')
    num_deno_classes: int = 41
    num_cono_classes: int = 2

    input_dir: str = new_base_path + 'data/processed/bill_mentions'
    # num_deno_classes: int = 1029

    # input_dir: str = '../data/processed/bill_mentions/title_deno_context5'
    # num_deno_classes: int = 1027

    # input_dir: str = '../data/processed/bill_mentions/title_deno'

    output_dir: Path = Path('../results/debug')
    #device: torch.device = torch.device('cuda')
    device: torch.device = torch.device('cpu')
    debug_subset_corpus: Optional[int] = None
    # dev_holdout: int = 5_000
    # test_holdout: int = 10_000
    num_dataloader_threads: int = 0
    pin_memory: bool = True

    delta: Optional[float] = None  # placeholders, assigned programmatically
    gamma: Optional[float] = None

    # Denotation Decomposer
    deno_size: int = 300
    deno_delta: float = 1  # denotation weight 𝛿
    deno_gamma: float = -1  # connotation weight 𝛾

    # Conotation Decomposer
    cono_size: int = 300
    cono_delta: float = -1  # denotation weight 𝛿
    cono_gamma: float = 1  # connotation weight 𝛾

    # Recomposer
    recomposer_rho: float = 1
    dropout_p: float = 0.1

    architecture: str = 'L4'
    batch_size: int = 128
    embed_size: int = 300
    num_epochs: int = 50
    encoder_update_cycle: int = 1  # per batch
    decoder_update_cycle: int = 1  # per batch

    pretrained_embedding: Optional[Path] = Path(new_base_path + 'data/pretrained_word2vec/for_real_SGNS.txt')
    freeze_embedding: bool = False
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
            '-i', '--input-dir', action='store', type=Path)
        parser.add_argument(
            '-o', '--output-dir', action='store', type=Path)
        parser.add_argument(
            '-gpu', '--device', action='store', type=str)

        parser.add_argument(
            '-a', '--architecture', action='store', type=str)
        parser.add_argument(
            '-dd', '--deno-delta', action='store', type=float)
        parser.add_argument(
            '-dg', '--deno-gamma', action='store', type=float)
        parser.add_argument(
            '-cd', '--cono-delta', action='store', type=float)
        parser.add_argument(
            '-cg', '--cono-gamma', action='store', type=float)

        parser.add_argument(
            '-lr', '--learning-rate', action='store', type=float)
        parser.add_argument(
            '-bs', '--batch-size', action='store', type=int)
        parser.add_argument(
            '-ep', '--num-epochs', action='store', type=int)
        parser.add_argument(
            '-pe', '--pretrained-embedding', action='store', type=Path)
        parser.parse_args(namespace=self)

        if self.architecture == 'L1R':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, self.num_deno_classes),
                nn.ReLU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, self.num_cono_classes),
                nn.ReLU())
        elif self.architecture == 'L2R':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, self.num_deno_classes),
                nn.ReLU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, self.num_cono_classes),
                nn.ReLU())
        elif self.architecture == 'L3R':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, self.num_deno_classes),
                nn.ReLU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, self.num_cono_classes),
                nn.ReLU())
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
                nn.Linear(300, self.num_deno_classes),
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
                nn.Linear(300, self.num_cono_classes),
                nn.SELU())
        elif self.architecture == 'L4R':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, self.num_deno_classes))
            self.cono_architecture = nn.Sequential(
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
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.num_deno_classes))
            self.cono_architecture = nn.Sequential(
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


def main() -> None:
    config = RecomposerConfig()
    black_box = RecomposerExperiment(config)
    #with black_box as auto_save_wrapped:
    #    auto_save_wrapped.train()
    p_embedding = black_box.model.deno_decomposer.embedding.weight.detach().cpu().numpy()
    print("pretrained embedding shape", p_embedding.shape)
    w2id = black_box.model.word_to_id
    id2w = black_box.model.id_to_word

    # Experiment control knobs
    experiment_name = "sort" # One of: cono, pos, sort
    use_avg_prec = False
    transform_embeddings = True
    use_pca = True # Otherwise, ICA. Only applies if transform_embeddings is True.
    
    if experiment_name == "pos":
        master_pos_dict = get_full_pos_dict()
        global_pos_list_l = list(global_pos_list)
        pos_one_hot = [[] for i in global_pos_list_l]
    
    ground = black_box.model.deno_decomposer.grounding
    query_conos = []
    filtered_embeddings = []
    found_count = 0
    
    deno_choices = black_box.model.deno_decomposer.deno_to_id.keys()
    
    query_denos = {d:[] for d in deno_choices}
    
    print("number of query words", len(w2id.keys()))
    
    
    for query_word in w2id.keys():
    
        id = w2id[query_word]
    
        if experiment_name == "cono" or experiment_name == "sort":
            #if "_" not in query_word:
            #    # Only use compound words
            #    continue

            query_cono = ground[query_word]['R_ratio']
            query_conos.append(query_cono)
            query_deno = ground[query_word]['majority_deno']
            
            for deno in deno_choices:
                if deno == query_deno:
                    query_denos[deno].append(1)
                else:
                    query_denos[deno].append(0)

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
    
    print("pos found", found_count)
    print("global_pos_list", global_pos_list)
    
    unfiltered_embeddings = filtered_embeddings
    if transform_embeddings:
        if use_pca:
            ca = PCA()
        else:
            ca = FastICA()
        ca.fit(filtered_embeddings)
        
        filtered_embeddings = ca.transform(filtered_embeddings)

    if experiment_name == "cono":
        rvals = []
        for emb_pos in range(filtered_embeddings.shape[1]):
            rval, ptail = stats.spearmanr(query_conos, filtered_embeddings[:,emb_pos])
            rvals.append((rval, emb_pos, ptail))
    
        rvals.sort()
        print("min, max cono corr:", rvals[0], rvals[-1])

        for deno in deno_choices:
            rvals = []
            for emb_pos in range(filtered_embeddings.shape[1]):
                if use_avg_prec:
                    avg_prec_pos = average_precision_score(query_denos[deno], filtered_embeddings[:,emb_pos])
                    rvals.append((avg_prec_pos, emb_pos))
                else:
                    rval, pval = stats.pointbiserialr(query_denos[deno], filtered_embeddings[:,emb_pos])
                    rvals.append((rval, emb_pos))
            rvals.sort()
            if use_avg_prec:
                print("avg prec. deno corr ({:45s})".format(deno), rvals[-1])
            else:
                print("min, max deno corr ({:45s})".format(deno), rvals[0], rvals[-1])

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
    
    else: # "sort"
        
        num_sort_components = 8 # Start at the left (might not be principal)
        
        sorted_idx = np.argsort(filtered_embeddings,axis=0)
        
        max_word_len = len(max(w2id.keys(), key=len))
        
        print("max word len", max_word_len)
        
        #for r in sorted_idx:
        #    for c in r[:num_sort_components]:
        #        sys.stdout.write("{:20.20s} ".format(id2w[c]))
        #    print("")
        
        target_idx = 1 # PCA component offset
        display_len = 3000
        display_step = 20
        
        #idces = sorted_idx[0:display_len:display_step,target_idx]
        idces = sorted_idx[:,target_idx]
        
        fig = plt.figure()
        if True: # Experiment 1 i.e. one-hot vector
            target_deno = 'Immigration'
            #component0vals = filtered_embeddings[idces,target_idx]
            component0vals = np.array(query_denos[target_deno])[idces]
            ax = plt.axes()
            N = 50 # Window sizse
            ax.scatter(x=range(len(component0vals)), y=np.convolve(component0vals, np.ones((N,))/N, mode='same'))
            ax.set_xlabel('Word ID')
            ax.set_ylabel('Moving average of one-hot denotation vector')
        else: #3D plot
        
            tsne_proj_eng = TSNE(n_components=3, random_state=0)
            tsne_data = tsne_proj_eng.fit_transform(unfiltered_embeddings[idces[:display_len],:])
        
            cmhot = plt.get_cmap("RdYlGn")
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(tsne_data[:,0], 
                       tsne_data[:,1],
                       tsne_data[:,2],
                       c=filtered_embeddings[idces[:display_len],target_idx], 
                       cmap=cmhot)
        plt.show()

if __name__ == '__main__':
    main()
