import argparse
import random
from copy import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from decomposer import Decomposer, DecomposerConfig, LabeledSentences
from utils.experiment import Experiment
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor

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

        dev_path = Path('../data/ellie/partisan_sample_val.cr.txt')
        with open(dev_path) as file:
            self.dev_ids = torch.tensor(
                [self.word_to_id[word.strip()] for word in file],
                device=self.device)

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

                    DS_Hdeno, DS_Hcono, CS_Hdeno, CS_Hcono = model.homemade_homogeneity(model.dev_ids)
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
    input_dir: Path = Path('../data/processed/bill_mentions/topic_deno')
    num_deno_classes: int = 41
    num_cono_classes: int = 2

    # input_dir: str = '../data/processed/bill_mentions/title_deno'

    output_dir: Path = Path('../results/debug')
    device: torch.device = torch.device('cuda')
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

    pretrained_embedding: Optional[Path] = Path('../data/pretrained_word2vec/bill_mentions_SGNS.txt')
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

        if self.architecture == 'L1':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, self.num_deno_classes),
                nn.SELU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, self.num_cono_classes),
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
                nn.Linear(300, self.num_cono_classes),
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
        else:
            raise ValueError('Unknown architecture argument.')


def main() -> None:
    config = RecomposerConfig()
    black_box = RecomposerExperiment(config)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    main()
