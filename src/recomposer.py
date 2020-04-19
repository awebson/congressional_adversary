import argparse
import random
from copy import copy
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from decomposer import Decomposer, DecomposerConfig, LabeledDocuments
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
        self.beta = config.beta
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
        if config.pretrained_embedding is not None:
            config.embed_size = data.pretrained_embedding.shape[1]
            self.pretrained_embed = nn.Embedding.from_pretrained(data.pretrained_embedding)
            self.pretrained_embed.weight.requires_grad = False
        else:
            raise ValueError('Reconstruction loss requires a pretrained embedding.')
        self.recomposer = nn.Linear(600, 300)
        self.rho = config.recomposer_rho
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

        recomposed = self.recomposer(torch.cat((deno_vecs, cono_vecs), dim=-1))
        # recomposed = deno_vecs + cono_vecs  # cosine similarity ignores magnitude
        pretrained = self.pretrained_embed(seq_word_ids)
        L_R = 1 - nn.functional.cosine_similarity(recomposed, pretrained, dim=-1).mean()

        # L_D += self.rho * L_R
        # L_C += self.rho * L_R
        L_joint = L_D + L_C + self.rho * L_R
        # return L_D, l_Dd, l_Dc, L_C, l_Cd, l_Cc, L_R
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

    def NN_cluster_homogeneity(
            self,
            query_ids: Vector,
            top_k: int = 5
            ) -> Tuple[float, ...]:
        D_homogeneity, D_homemade_homogeneity = self.deno_decomposer.NN_cluster_homogeneity(
            query_ids, eval_deno=True, top_k=top_k)
        C_homogeneity, C_homemade_homogeneity = self.cono_decomposer.NN_cluster_homogeneity(
            query_ids, eval_deno=False, top_k=top_k)
        return D_homogeneity, D_homemade_homogeneity, C_homogeneity, C_homemade_homogeneity


class RecomposerExperiment(Experiment):

    def __init__(self, config: 'RecomposerConfig'):
        super().__init__(config)
        self.data = LabeledDocuments(config)
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
        # self.D_deno_optimizer = config.optimizer(
        #     self.model.deno_decomposer.deno_decoder.parameters(),
        #     lr=config.learning_rate)
        self.D_cono_optimizer = config.optimizer(
            self.model.deno_decomposer.cono_decoder.parameters(),
            lr=config.learning_rate)

        self.C_decomp_optimizer = config.optimizer(
            self.model.cono_decomposer.embedding.parameters(),
            lr=config.learning_rate)
        # self.C_deno_optimizer = config.optimizer(
        #     self.model.cono_decomposer.deno_decoder.parameters(),
        #     lr=config.learning_rate)
        self.C_cono_optimizer = config.optimizer(
            self.model.cono_decomposer.cono_decoder.parameters(),
            lr=config.learning_rate)

        self.R_optimizer = config.optimizer(
            self.model.recomposer.parameters(),
            lr=config.learning_rate)

        self.to_be_saved = {
            'config': self.config,
            'model': self.model}

        from evaluations.helpers import polarized_words
        self.eval_word_ids = torch.tensor(
            [w.word_id for w in polarized_words], device=self.device)

    def _train(
            self,
            center_word_ids: Vector,
            context_word_ids: Vector,
            seq_word_ids: Matrix,
            cono_labels: Vector,
            ) -> Tuple[float, ...]:
        grad_clip = self.config.clip_grad_norm
        self.model.zero_grad()
        L_D, l_Dd, l_Dc, L_C, l_Cd, l_Cc, L_R, L_joint = self.model(
            center_word_ids, context_word_ids,
            seq_word_ids, cono_labels)

        # Denotation Decomposer
        L_joint.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model.deno_decomposer.embedding.parameters(), grad_clip)
        self.D_decomp_optimizer.step()

        # self.model.zero_grad()
        # l_Dd.backward(retain_graph=True)
        # self.D_deno_optimizer.step()

        self.model.zero_grad()
        l_Dc.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model.deno_decomposer.cono_decoder.parameters(), grad_clip)
        self.D_cono_optimizer.step()

        # Connotation Decomposer
        self.model.zero_grad()
        L_joint.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model.deno_decomposer.embedding.parameters(), grad_clip)
        self.C_decomp_optimizer.step()

        # self.model.zero_grad()
        # l_Cd.backward(retain_graph=True)
        # self.C_deno_optimizer.step()

        self.model.zero_grad()
        l_Cc.backward(retain_graph=True)  # NOTE
        nn.utils.clip_grad_norm_(self.model.cono_decomposer.cono_decoder.parameters(), grad_clip)
        self.C_cono_optimizer.step()

        # Recomposer
        self.model.zero_grad()
        L_R.backward()
        nn.utils.clip_grad_norm_(self.model.recomposer.parameters(), grad_clip)
        self.R_optimizer.step()

        return (
            L_D.item(),
            l_Dd.item(),
            l_Dc.item(),
            L_C.item(),
            l_Cd.item(),
            l_Cc.item(),
            L_R.item(),
            L_joint.item())

    def train(self) -> None:
        config = self.config
        model = self.model
        # For debugging
        # self.save_everything(
        #     os.path.join(self.config.output_dir, f'untrained.pt'))
        # import IPython
        # IPython.embed()
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

                L_D, l_Dd, l_Dc, L_C, l_Cd, l_Cc, L_R, L_joint = self._train(
                    center_word_ids, context_word_ids,  # negative_ids,
                    seq_word_ids, cono_labels)

                if batch_index % config.update_tensorboard == 0:
                    D_cono_acc, C_cono_acc = model.accuracy(
                        seq_word_ids, cono_labels)
                    self.update_tensorboard({
                        'Denotation Decomposer/deno_loss': l_Dd,
                        'Denotation Decomposer/cono_loss': l_Dc,
                        # 'Denotation Decomposer/accuracy_train_deno': D_deno_acc,
                        'Denotation Decomposer/accuracy_train_cono': D_cono_acc,

                        'Connotation Decomposer/deno_loss': l_Cd,
                        'Connotation Decomposer/cono_loss': l_Cc,
                        # 'Connotation Decomposer/accuracy_train_deno': C_deno_acc,
                        'Connotation Decomposer/accuracy_train_cono': C_cono_acc,

                        # 'Combined Losses/Denotation Decomposer': L_D,
                        # 'Combined Losses/Connotation Decomposer': L_C,
                        'Joint/loss': L_joint,
                        'Joint/Recomposer': L_R
                    })

                # if batch_index % config.eval_dev_set == 0:
                #     D_deno_acc, D_cono_acc, C_deno_acc, C_cono_acc = model.accuracy(
                #         self.data.dev_seq.to(self.device),
                #         self.data.dev_deno_labels.to(self.device),
                #         self.data.dev_cono_labels.to(self.device))

                #     D_h, D_hh, C_h, C_hh = model.NN_cluster_homogeneity(self.eval_word_ids)
                #     self.update_tensorboard({
                #         'Denotation Decomposer/accuracy_dev_deno': D_deno_acc,
                #         'Denotation Decomposer/accuracy_dev_cono': D_cono_acc,
                #         'Connotation Decomposer/accuracy_dev_deno': C_deno_acc,
                #         'Connotation Decomposer/accuracy_dev_cono': C_cono_acc,

                #         'Denotation Decomposer/homogeneity': D_h,
                #         'Denotation Decomposer/homogeneity homemade': D_hh,
                #         'Connotation Decomposer/homogeneity': C_h,
                #         'Connotation Decomposer/homogeneity homemade': C_hh,
                #     })
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
    input_dir: str = '../data/processed/labeled_documents/for_real'
    output_dir: str = '../results/debug'
    device: torch.device = torch.device('cuda')
    debug_subset_corpus: Optional[int] = None
    # dev_holdout: int = 5_000
    # test_holdout: int = 10_000
    num_dataloader_threads: int = 6
    pin_memory: bool = True

    delta: Optional[float] = None  # placeholders, assigned programmatically
    gamma: Optional[float] = None
    beta: float = 10  # bias term
    # Denotation Decomposer
    deno_size: int = 300
    deno_delta: float = 1  # denotation weight 𝛿
    deno_gamma: float = -10  # connotation weight 𝛾

    # Conotation Decomposer
    cono_size: int = 300
    cono_delta: float = -0.1  # denotation weight 𝛿
    cono_gamma: float = 10  # connotation weight 𝛾

    # Recomposer
    recomposer_rho: float = 10
    dropout_p: float = 0

    architecture: str = 'L1'
    batch_size: int = 128
    embed_size: int = 300
    num_epochs: int = 10
    encoder_update_cycle: int = 1  # per batch
    decoder_update_cycle: int = 1  # per batch

    pretrained_embedding: Optional[str] = '../data/pretrained_word2vec/for_real.txt'
    freeze_embedding: bool = False  # NOTE
    window_radius: int = 5
    num_negative_samples: int = 10
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    learning_rate: float = 1e-4
    # momentum: float = 0.5
    # lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    # num_prediction_classes: int = 5
    clip_grad_norm: float = 5.0

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
    auto_save_per_epoch: Optional[int] = 2
    auto_save_if_interrupted: bool = False

    def __post_init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-i', '--input-dir', action='store', type=str)
        parser.add_argument(
            '-o', '--output-dir', action='store', type=str)
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
            '-pe', '--pretrained-embedding', action='store', type=str)
        parser.parse_args(namespace=self)

        if self.architecture == 'L1':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 41),
                nn.SELU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 2),
                nn.SELU())
        elif self.architecture == 'L2':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 41),
                nn.SELU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 2),
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
                nn.Linear(300, 41),
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
                nn.Linear(300, 2),
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
