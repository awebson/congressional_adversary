import argparse
import random
from copy import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from decomposer import Decomposer, DecomposerConfig, LabeledDocuments
from data import Sentence, LabeledDoc, GroundedWord
from evaluations.word_similarity import all_wordsim as word_sim
from utils.experiment import Experiment
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor

random.seed(42)
torch.manual_seed(42)

class Recomposer(nn.Module):

    def __init__(
            self,
            config: 'RecomposerConfig',
            data: 'LabeledDocuments'):
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
        # self.recomposer = nn.Linear(600, 300)
        self.rho = config.rho
        self.to(self.device)

        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        # self.grounding = data.grounding

    def forward(
            self,
            center_word_ids: Vector,
            context_word_ids: Vector,
            seq_word_ids: Matrix,
            cono_labels: Vector,
            ) -> Tuple[Scalar, ...]:
        L_D, l_Dd, l_Dc, deno_vecs = self.deno_decomposer(
            center_word_ids, context_word_ids,
            seq_word_ids, cono_labels, recompose=True)
        L_C, l_Cd, l_Cc, cono_vecs = self.cono_decomposer(
            center_word_ids, context_word_ids,
            seq_word_ids, cono_labels, recompose=True)

        # recomposed = self.recomposer(torch.cat((deno_vecs, cono_vecs), dim=-1))
        recomposed = deno_vecs + cono_vecs  # cosine similarity ignores magnitude
        pretrained = self.deno_decomposer.pretrained_embed(seq_word_ids)
        L_R = 1 - F.cosine_similarity(recomposed, pretrained, dim=-1).mean()

        L_joint = L_D + L_C + self.rho * L_R
        return L_D, l_Dd, l_Dc, L_C, l_Cd, l_Cc, L_R, L_joint

    def predict(self, seq_word_ids: Vector) -> Tuple[Vector, ...]:
        self.eval()
        D_cono_conf = self.deno_decomposer.predict(seq_word_ids)
        C_cono_conf = self.cono_decomposer.predict(seq_word_ids)
        self.train()
        return D_cono_conf, C_cono_conf

    def accuracy(
            self,
            seq_word_ids: Matrix,
            # deno_labels: Vector,
            cono_labels: Vector,
            error_analysis_path: Optional[str] = None
            ) -> Tuple[float, ...]:
        D_cono_accuracy = self.deno_decomposer.accuracy(
            seq_word_ids, cono_labels)
        C_cono_accuracy = self.cono_decomposer.accuracy(
            seq_word_ids, cono_labels)
        return D_cono_accuracy, C_cono_accuracy

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
        self.model = Recomposer(config, self.data)

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)  # param.data)

        # self.D_decomp_params = self.model.deno_decomposer.embedding.parameters()
        self.D_decomp_optimizer = config.optimizer(
            self.model.deno_decomposer.embedding.parameters(), lr=config.learning_rate)
        # self.D_cono_params = self.model.deno_decomposer.cono_decoder.parameters()
        self.D_cono_optimizer = config.optimizer(
            self.model.deno_decomposer.cono_decoder.parameters(), lr=config.learning_rate)
        # self.D_deno_optimizer = config.optimizer(
        #     self.model.deno_decomposer.deno_decoder.parameters(),
        #     lr=config.learning_rate)

        # self.C_decomp_params = self.model.cono_decomposer.embedding.parameters()
        self.C_decomp_optimizer = config.optimizer(
            self.model.cono_decomposer.embedding.parameters(), lr=config.learning_rate)
        # self.C_cono_parms = self.model.cono_decomposer.cono_decoder.parameters()
        self.C_cono_optimizer = config.optimizer(
            self.model.cono_decomposer.cono_decoder.parameters(), lr=config.learning_rate)
        # self.C_deno_optimizer = config.optimizer(
        #     self.model.cono_decomposer.deno_decoder.parameters(),
        #     lr=config.learning_rate)

        # self.recomp_params = self.model.recomposer.parameters()
        # self.R_optimizer = config.optimizer(self.recomp_params, lr=config.learning_rate)

        self.to_be_saved = {
            'config': self.config,
            'model': self.model}

    def train(self) -> None:
        config = self.config
        grad_clip = config.clip_grad_norm
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
                    total=self.data.estimated_len,
                    mininterval=config.progress_bar_refresh_rate,
                    desc='Batches')
            for batch_index, batch in batches:
                seq_word_ids = batch[0].to(self.device)
                center_word_ids = batch[1].to(self.device)
                context_word_ids = batch[2].to(self.device)
                cono_labels = batch[3].to(self.device)

                self.model.zero_grad()
                L_D, l_Dd, l_Dc, L_C, l_Cd, l_Cc, L_R, L_joint = self.model(
                    center_word_ids, context_word_ids,
                    seq_word_ids, cono_labels)

                # Denotation Decomposer
                L_joint.backward()
                nn.utils.clip_grad_norm_(model.deno_decomposer.embedding.parameters(), grad_clip)
                self.D_decomp_optimizer.step()

                nn.utils.clip_grad_norm_(model.deno_decomposer.cono_decoder.parameters(), grad_clip)
                self.D_cono_optimizer.step()

                # Connotation Decomposer
                nn.utils.clip_grad_norm_(model.cono_decomposer.embedding.parameters(), grad_clip)
                self.C_decomp_optimizer.step()

                nn.utils.clip_grad_norm_(model.cono_decomposer.cono_decoder.parameters(), grad_clip)
                self.C_cono_optimizer.step()

                # # Recomposer
                # nn.utils.clip_grad_norm_(self.recomp_params, grad_clip)
                # self.R_optimizer.step()

                if batch_index % config.update_tensorboard == 0:
                    D_cono_acc, C_cono_acc = model.accuracy(
                        seq_word_ids, cono_labels)
                    self.update_tensorboard({
                        'Denotation Decomposer/deno_loss': l_Dd,
                        'Denotation Decomposer/cono_loss': l_Dc,
                        'Denotation Decomposer/combined_loss': L_D,
                        'Denotation Decomposer/accuracy_train_cono': D_cono_acc,

                        'Connotation Decomposer/deno_loss': l_Cd,
                        'Connotation Decomposer/cono_loss': l_Cc,
                        'Connotation Decomposer/combined_loss': L_C,
                        'Connotation Decomposer/accuracy_train_cono': C_cono_acc,

                        'Joint/loss': L_joint,
                        'Joint/Recomposer': L_R
                    })
                if batch_index % config.eval_dev_set == 0:
                    self.validation()

                self.tb_global_step += 1
            # End Batches
            # self.lr_scheduler.step()
            self.data.estimated_len = batch_index
            self.print_timestamp(epoch_index)
            self.auto_save(epoch_index)
        # End Epochs

    def validation(self) -> None:
        # D_deno_acc, D_cono_acc, C_deno_acc, C_cono_acc = self.model.accuracy(
        #     self.data.dev_seq.to(self.device),
        #     self.data.dev_deno_labels.to(self.device),
        #     self.data.dev_cono_labels.to(self.device))
        model = self.model
        D_model = model.deno_decomposer
        DS_Hdeno, DS_Hcono = D_model.homemade_homogeneity(D_model.dev_ids)
        _, DS_Hcono_SP = D_model.SciPy_homogeneity(D_model.dev_ids)

        mean_delta, abs_rhos = word_sim.mean_delta(
            D_model.embedding.weight, D_model.pretrained_embed.weight,
            model.id_to_word, reduce=False)
        cos_sim = F.cosine_similarity(
            D_model.embedding.weight, D_model.pretrained_embed.weight).mean()
        self.update_tensorboard({
            'Denotation Space/Neighbor Overlap': DS_Hdeno,
            'Denotation Space/Party Homogeneity': DS_Hcono,
            'Denotation Space/Party Homogeneity SciPy': DS_Hcono_SP,
            'Denotation Space/Overlap - Party': DS_Hdeno - DS_Hcono,

            'Denotation Space/rho difference cf pretrained': mean_delta,
            'Denotation Space/MTurk-771': abs_rhos[0],
            'Denotation Space/cosine cf pretrained': cos_sim
        })

        C_model = model.cono_decomposer
        CS_Hdeno, CS_Hcono = C_model.homemade_homogeneity(C_model.dev_ids)
        _, CS_Hcono_SP = C_model.SciPy_homogeneity(C_model.dev_ids)
        mean_delta, abs_rhos = word_sim.mean_delta(
            C_model.embedding.weight, C_model.pretrained_embed.weight,
            model.id_to_word, reduce=False)
        cos_sim = F.cosine_similarity(
            C_model.embedding.weight, C_model.pretrained_embed.weight).mean()
        self.update_tensorboard({
            'Connotation Space/Neighbor Overlap': CS_Hdeno,
            'Connotation Space/Party Homogeneity': CS_Hcono,
            'Connotation Space/Party Homogeneity SciPy': CS_Hcono_SP,
            'Connotation Space/Party - Overlap': CS_Hcono - CS_Hdeno,

            'Connotation Space/rho difference cf pretrained': mean_delta,
            'Connotation Space/MTurk-771': abs_rhos[0],
            'Connotation Space/cosine cf pretrained': cos_sim
        })

        with torch.no_grad():
            # sample = torch.randint(
            #     D_model.embedding.num_embeddings, size=(25_000,), device=self.device)
            sample = torch.arange(D_model.embedding.num_embeddings, device=self.device)
            recomposed = D_model.embedding(sample) + C_model.embedding(sample)

        mean_delta, abs_rhos = word_sim.mean_delta(
            recomposed,
            D_model.pretrained_embed.weight,
            model.id_to_word,
            reduce=False)
        self.update_tensorboard({
            'Recomposer/mean IntraSpace quality': ((DS_Hdeno - DS_Hcono) + (CS_Hcono - CS_Hdeno)) / 2,

            'Recomposer/rho difference cf pretrained': mean_delta,
            'Recomposer/MTurk-771': abs_rhos[0],
            'Recomposer/cosine similarity':
                F.cosine_similarity(recomposed, D_model.pretrained_embed(sample), dim=1).mean()
        })


@dataclass
class RecomposerConfig():
    # Essential
    input_dir: Path = Path('../data/ready/3bin')
    output_dir: Path = Path('../results/debug')
    device: torch.device = torch.device('cuda')
    debug_subset_corpus: Optional[int] = None
    # dev_holdout: int = 5_000
    # test_holdout: int = 10_000
    num_dataloader_threads: int = 0
    pin_memory: bool = True

    # placeholders, assigned programmatically
    delta: Optional[float] = None
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
    rho: float = 1
    dropout_p: float = 0.1

    architecture: str = 'L4'
    batch_size: int = 1024
    embed_size: int = 300
    num_epochs: int = 10

    pretrained_embedding: Optional[Path] = Path('../data/pretrained_word2vec/partisan_news.txt')
    freeze_embedding: bool = False  # NOTE
    skip_gram_window_radius: int = 5
    num_negative_samples: int = 10
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    learning_rate: float = 1e-4
    # momentum: float = 0.5
    # lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    # num_prediction_classes: int = 5
    clip_grad_norm: float = 10.0

    # Housekeeping
    export_error_analysis: Optional[int] = 1  # per epoch
    update_tensorboard: int = 1000  # per batch
    print_stats: Optional[int] = 10_000  # per batch
    eval_dev_set: int = 100_000  # per batch  # NOTE
    progress_bar_refresh_rate: int = 1  # per second
    suppress_stdout: bool = False  # during hyperparameter tuning
    reload_path: Optional[str] = None
    clear_tensorboard_log_in_output_dir: bool = True
    delete_all_exisiting_files_in_output_dir: bool = False
    auto_save_per_epoch: Optional[int] = 1
    auto_save_if_interrupted: bool = True

    def __post_init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-i', '--input-dir', action='store', type=Path)
        parser.add_argument(
            '-o', '--output-dir', action='store', type=Path)
        parser.add_argument(
            '-gpu', '--device', action='store', type=str)

        parser.add_argument(
            '-dd', '--deno-delta', action='store', type=float)
        parser.add_argument(
            '-dg', '--deno-gamma', action='store', type=float)
        parser.add_argument(
            '-cd', '--cono-delta', action='store', type=float)
        parser.add_argument(
            '-cg', '--cono-gamma', action='store', type=float)

        parser.add_argument(
            '-a', '--architecture', action='store', type=str)
        parser.add_argument(
            '-lr', '--learning-rate', action='store', type=float)
        parser.add_argument(
            '-bs', '--batch-size', action='store', type=int)
        parser.add_argument(
            '-ep', '--num-epochs', action='store', type=int)
        parser.parse_args(namespace=self)

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
    config = RecomposerConfig()
    black_box = RecomposerExperiment(config)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    main()
