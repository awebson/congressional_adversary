import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Counter, Optional, Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.dual_grounded import Decomposer, Recomposer
from utils.experiment import Experiment
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor
from evaluations.word_similarity import all_wordsim as word_sim


class ProxyGroundedDecomposer(Decomposer):

    def __init__(
            self,
            preserve: str,  # either 'deno' or 'cono'
            initial_space: Matrix,
            cono_probe: nn.Module,
            id_to_word: Dict[int, str],
            grounding: Dict[str, Dict[str, Any]],
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

        # Initailize neighbor cono homogeneity eval partisan vocabulary
        self.id_to_word = id_to_word
        self.grounding = grounding
        # self.init_grounding()

    # def init_grounding(self) -> None:
    #     dev_path = Path('../data/ellie/partisan_sample_val.cr.txt')  # TODO renew sampling
    #     with open(dev_path) as file:
    #         self.dev_ids = torch.tensor(
    #             [self.word_to_id[word.strip()] for word in file],
    #             device=self.device)

    #     def pretrained_neighbors(
    #             query_ids: Vector,
    #             top_k: int = 10
    #             ) -> Dict[int, Set[int]]:
    #         deno_grounding: Dict[int, Set[int]] = {}
    #         # self.pretrained_embed = self.pretrained_embed.to(self.device)
    #         # with torch.no_grad():
    #         for qid in query_ids:
    #             qv = self.pretrained_embed(qid)
    #             qid = qid.item()
    #             qw = self.id_to_word[qid]
    #             cos_sim = F.cosine_similarity(qv.unsqueeze(0), self.pretrained_embed.weight)
    #             cos_sim, neighbor_ids = cos_sim.topk(k=top_k + 5, dim=-1)
    #             neighbor_ids = [
    #                 nid for nid in neighbor_ids.tolist()
    #                 if editdistance.eval(qw, self.id_to_word[nid]) > 3]
    #             deno_grounding[qid] = set(neighbor_ids[:top_k])
    #         return deno_grounding

    #     self.deno_grounding: Dict[int, Set[int]] = pretrained_neighbors(self.dev_ids)

    def forward(
            self,
            center_word_ids: Vector,
            true_context_ids: Vector,
            seq_word_ids: Matrix,
            cono_labels: Vector,
            ) -> Tuple[Scalar, ...]:
        seq_word_vecs: R3Tensor = self.embedding(seq_word_ids)
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
            cono = self.cono_probe(seq_repr)
            cono_conf = nn.functional.softmax(cono, dim=1)
        self.train()
        return cono_conf

    def accuracy(
            self,
            seq_word_ids: Matrix,
            cono_labels: Vector,
            error_analysis_path: Optional[str] = None
            ) -> float:
        cono_conf = self.predict(seq_word_ids)
        cono_predictions = cono_conf.argmax(dim=1)
        cono_correct_indicies = cono_predictions.eq(cono_labels)
        cono_accuracy = cono_correct_indicies.float().mean().item()
        return cono_accuracy


class ProxyGroundedRecomposer(Recomposer):

    def __init__(
            self,
            config: 'RecomposerConfig',
            data: 'LabeledDocuments'):
        super(Recomposer, self).__init__()
        self.device = config.device

        self.pretrained_embed = Experiment.load_txt_embedding(
            config.pretrained_embed_path, data.word_to_id)
        self.pretrained_embed.weight.requires_grad = False

        self.deno_space = Decomposer(
            preserve='deno',
            initial_space=self.pretrained_embed.weight,
            cono_probe=config.cono_probe,
            id_to_word=data.id_to_word,
            grounding=data.grounding,
            num_negative_samples=config.num_negative_samples,
            negative_sampling_probs=data.negative_sampling_probs,
            device=self.device)

        self.cono_space = Decomposer(
            preserve='cono',
            initial_space=self.pretrained_embed.weight,
            cono_probe=config.cono_probe,
            id_to_word=data.id_to_word,
            grounding=data.grounding,
            num_negative_samples=config.num_negative_samples,
            negative_sampling_probs=data.negative_sampling_probs,
            device=self.device)
        assert config.cono_probe[-1].out_features == config.num_cono_classes

        # Recomposer
        # self.recomposer = nn.Linear(600, 300)
        self.rho = config.rho
        self.to(self.device)

        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.grounding = data.grounding

    def forward(
            self,
            center_word_ids: Vector,
            context_word_ids: Vector,
            seq_word_ids: Matrix,
            cono_labels: Vector,
            ) -> Tuple[Scalar, ...]:
        L_D, l_Dd, l_Dcp, l_Dca, deno_vecs = self.deno_space(
            center_word_ids, context_word_ids,
            seq_word_ids, cono_labels)
        L_C, l_Cd, l_Ccp, l_Cca, cono_vecs = self.cono_space(
            center_word_ids, context_word_ids,
            seq_word_ids, cono_labels)

        # recomposed = self.recomposer(torch.cat((deno_vecs, cono_vecs), dim=-1))
        recomposed = deno_vecs + cono_vecs  # cosine similarity ignores magnitude
        pretrained = self.deno_space.pretrained_embed(seq_word_ids)
        L_R = 1 - F.cosine_similarity(recomposed, pretrained, dim=-1).mean()

        L_joint = L_D + L_C + self.rho * L_R
        return L_D, l_Dd, l_Dcp, l_Dca, L_C, l_Cd, l_Ccp, l_Cca, L_R, L_joint

    def predict(self, seq_word_ids: Vector) -> Tuple[Vector, ...]:
        self.eval()
        D_cono_conf = self.deno_space.predict(seq_word_ids)
        C_cono_conf = self.cono_space.predict(seq_word_ids)
        self.train()
        return D_cono_conf, C_cono_conf

    def accuracy(
            self,
            seq_word_ids: Matrix,
            # deno_labels: Vector,
            cono_labels: Vector,
            error_analysis_path: Optional[str] = None
            ) -> Tuple[float, ...]:
        D_cono_accuracy = self.deno_space.accuracy(
            seq_word_ids, cono_labels)
        C_cono_accuracy = self.cono_space.accuracy(
            seq_word_ids, cono_labels)
        return D_cono_accuracy, C_cono_accuracy


class LabeledDocuments(Dataset):

    def __init__(self, config: 'DecomposerConfig'):
        with open(corpus_path, 'rb') as corpus_file:
            preprocessed = pickle.load(corpus_file)
        self.word_to_id = preprocessed['word_to_id']
        self.id_to_word = preprocessed['id_to_word']
        self.documents: List[List[int]] = preprocessed['documents']
        self.cono_labels: List[int] = preprocessed['cono_labels']
        self.negative_sampling_probs = torch.tensor(preprocessed['negative_sampling_probs'])
        self.grounding: Dict[str, Counter[str]] = preprocessed['grounding']
        self.window_radius = config.skip_gram_window_radius
        self.num_negative_samples = config.num_negative_samples
        self.fixed_sent_len = 15  # HACK
        self.min_sent_len = 5

        # deno/cono -> Counter[word]
        # self.counts: Dict[str, Counter[str]] = preprocessed['counts']

        # self.train_seq: List[List[int]] = preprocessed['train_sent_word_ids']
        # self.train_cono_labels: List[int] = preprocessed['train_cono_labels']

        # self.dev_seq = rnn.pad_sequence(
        #     [torch.tensor(seq) for seq in preprocessed['dev_sent_word_ids']],
        #     batch_first=True)
        # self.dev_cono_labels = torch.tensor(preprocessed['dev_cono_labels'])

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(
            self,
            index: int
            ) -> Tuple[
            List[int],
            List[int],
            # List[List[int]],
            List[List[int]],
            int]:
        """process one document into skip-gram pairs and faux sentences"""
        doc: List[int] = self.documents[index]

        # For Denotation
        # parsing (center, context) word id pairs
        center_word_ids: List[int] = []
        context_word_ids: List[int] = []
        for center_index, center_word_id in enumerate(doc):
            left_index = max(center_index - self.window_radius, 0)
            right_index = min(center_index + self.window_radius, len(doc) - 1)
            context_word_id: List[int] = (
                doc[left_index:center_index] +
                doc[center_index + 1:right_index + 1])
            context_word_ids += context_word_id
            center_word_ids += [center_word_id] * len(context_word_id)
            # labels += [self.id_to_label[center_word_id]] * len(context_word_id)

        # negative_context_ids = torch.multinomial(
        #     self.negative_sampling_probs,
        #     len(context_word_ids) * self.num_negative_samples,
        #     replacement=True
        # ).view(len(context_word_ids), self.num_negative_samples)

        # For Connotation, split a document into faux sentences
        faux_sentences: List[List[int]] = []
        start_index = 0
        while (start_index + self.fixed_sent_len) < (len(doc) - 1):
            faux_sentences.append(
                doc[start_index:start_index + self.fixed_sent_len])
            start_index += self.fixed_sent_len

        trailing_words = doc[start_index:-1]
        if len(trailing_words) >= self.min_sent_len:
            faux_sentences.append(trailing_words)

        return (
            center_word_ids,
            context_word_ids,
            # negative_context_ids,
            faux_sentences,
            self.cono_labels[index])

    @staticmethod
    def collate(
            processed_docs: List[Tuple[
            List[int],
            List[int],
            # List[List[int]],
            List[List[int]],
            int]]
            ) -> Tuple[Vector, Vector, Matrix, Vector]:
        """flatten indeterminate docs full of data into batches"""
        # seq_word_ids = [torch.tensor(w) for w, _, _ in processed_docs]
        # deno_labels = torch.tensor([d for _, d, _ in processed_docs])
        # cono_labels = torch.tensor([c for _, _, c in processed_docs])
        center_ids = []
        context_ids = []
        # negative_ids = []
        seq_word_ids: List[Vector] = []
        cono_labels = []
        for center, context, seqs, cono_label in processed_docs:
            center_ids += center
            context_ids += context
            # negative_ids += neg_sample
            seq_word_ids += [torch.tensor(seq) for seq in seqs]
            cono_labels += [cono_label] * len(seqs)

        # import IPython
        # IPython.embed()
        return (
            torch.tensor(center_ids),
            torch.tensor(context_ids),
            # torch.tensor(negative_ids),
            rnn.pad_sequence(seq_word_ids, batch_first=True),
            torch.tensor(cono_labels))


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
            pin_memory=True)
        self.model = Recomposer(config, self.data)

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)  # param.data)

        # self.D_decomp_params = self.model.deno_space.embedding.parameters()
        self.D_decomp_optimizer = config.optimizer(
            self.model.deno_space.embedding.parameters(), lr=config.learning_rate)
        # self.D_cono_params = self.model.deno_space.cono_probe.parameters()
        self.D_cono_optimizer = config.optimizer(
            self.model.deno_space.cono_probe.parameters(), lr=config.learning_rate)
        # self.D_deno_optimizer = config.optimizer(
        #     self.model.deno_space.deno_decoder.parameters(),
        #     lr=config.learning_rate)

        # self.C_decomp_params = self.model.cono_space.embedding.parameters()
        self.C_decomp_optimizer = config.optimizer(
            self.model.cono_space.embedding.parameters(), lr=config.learning_rate)
        # self.C_cono_parms = self.model.cono_space.cono_probe.parameters()
        self.C_cono_optimizer = config.optimizer(
            self.model.cono_space.cono_probe.parameters(), lr=config.learning_rate)
        # self.C_deno_optimizer = config.optimizer(
        #     self.model.cono_space.deno_decoder.parameters(),
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
        # For debugging
        self.save_everything(self.config.output_dir / 'init_recomposer.pt')
        raise SystemExit

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
                center_word_ids = batch[0].to(self.device)
                context_word_ids = batch[1].to(self.device)
                seq_word_ids = batch[2].to(self.device)
                cono_labels = batch[3].to(self.device)

                self.model.zero_grad()
                L_D, l_Dd, l_Dcp, l_Dca, L_C, l_Cd, l_Ccp, l_Cca, L_R, L_joint = self.model(
                    center_word_ids, context_word_ids, seq_word_ids, cono_labels)

                # Denotation Decomposer
                L_joint.backward()
                nn.utils.clip_grad_norm_(model.deno_space.embedding.parameters(), grad_clip)
                self.D_decomp_optimizer.step()

                # Connotation Decomposer
                nn.utils.clip_grad_norm_(model.cono_space.embedding.parameters(), grad_clip)
                self.C_decomp_optimizer.step()

                self.model.zero_grad()
                L_D, l_Dd, l_Dcp, l_Dca, _ = self.model.deno_space(
                    center_word_ids, context_word_ids, seq_word_ids, cono_labels)
                l_Dcp.backward()
                nn.utils.clip_grad_norm_(model.deno_space.cono_probe.parameters(), grad_clip)
                self.D_cono_optimizer.step()

                self.model.zero_grad()
                L_C, l_Cd, l_Ccp, l_Cca, _ = self.model.cono_space(
                    center_word_ids, context_word_ids, seq_word_ids, cono_labels)
                l_Ccp.backward()
                nn.utils.clip_grad_norm_(model.cono_space.cono_probe.parameters(), grad_clip)
                self.C_cono_optimizer.step()

                # # Recomposer
                # nn.utils.clip_grad_norm_(self.recomp_params, grad_clip)
                # self.R_optimizer.step()

                if batch_index % config.update_tensorboard == 0:
                    D_cono_acc, C_cono_acc = model.accuracy(
                        seq_word_ids, cono_labels)
                    self.update_tensorboard({
                        'Denotation Decomposer/deno_loss': l_Dd,
                        'Denotation Decomposer/cono_loss_proper': l_Dcp,
                        'Denotation Decomposer/cono_loss_adversary': l_Dca,
                        'Denotation Decomposer/combined_loss': L_D,
                        'Denotation Decomposer/accuracy_train_cono': D_cono_acc,

                        'Connotation Decomposer/deno_loss': l_Cd,
                        'Connotation Decomposer/cono_loss_proper': l_Ccp,
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
            self.print_timestamp(epoch_index)
            self.auto_save(epoch_index)
        # End Epochs

    def validation(self) -> None:
        # D_deno_acc, D_cono_acc, C_deno_acc, C_cono_acc = self.model.accuracy(
        #     self.data.dev_seq.to(self.device),
        #     self.data.dev_deno_labels.to(self.device),
        #     self.data.dev_cono_labels.to(self.device))
        model = self.model
        D_model = model.deno_space
        DS_Hdeno, DS_Hcono = D_model.homemade_homogeneity(D_model.dev_ids)
        # _, DS_Hcono_SP = D_model.SciPy_homogeneity(D_model.dev_ids)

        mean_delta, abs_rhos = word_sim.mean_delta(
            D_model.embedding.weight, D_model.pretrained_embed.weight,
            model.id_to_word, reduce=False)
        cos_sim = F.cosine_similarity(
            D_model.embedding.weight, D_model.pretrained_embed.weight).mean()
        self.update_tensorboard({
            'Denotation Space/Neighbor Overlap': DS_Hdeno,
            'Denotation Space/Party Homogeneity': DS_Hcono,
            # 'Denotation Space/Party Homogeneity SciPy': DS_Hcono_SP,
            'Denotation Space/Overlap - Party': DS_Hdeno - DS_Hcono,

            'Denotation Space/rho difference cf pretrained': mean_delta,
            'Denotation Space/MTurk-771': abs_rhos[0],
            'Denotation Space/cosine cf pretrained': cos_sim
        })

        C_model = model.cono_space
        CS_Hdeno, CS_Hcono = C_model.homemade_homogeneity(C_model.dev_ids)
        # _, CS_Hcono_SP = C_model.SciPy_homogeneity(C_model.dev_ids)
        mean_delta, abs_rhos = word_sim.mean_delta(
            C_model.embedding.weight, C_model.pretrained_embed.weight,
            model.id_to_word, reduce=False)
        cos_sim = F.cosine_similarity(
            C_model.embedding.weight, C_model.pretrained_embed.weight).mean()
        self.update_tensorboard({
            'Connotation Space/Neighbor Overlap': CS_Hdeno,
            'Connotation Space/Party Homogeneity': CS_Hcono,
            # 'Connotation Space/Party Homogeneity SciPy': CS_Hcono_SP,
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
    input_dir: Path = Path('../data/processed/CR_skip')
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
    batch_size: int = 2
    embed_size: int = 300
    num_epochs: int = 10

    pretrained_embedding: Optional[Path] = Path('../data/pretrained_word2vec/for_real.txt')
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
    progress_bar_refresh_rate: int = 60  # per second
    suppress_stdout: bool = False  # during hyperparameter tuning
    reload_path: Optional[str] = None
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

        self.num_cono_classes = 2
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
        else:
            raise ValueError('Unknown architecture argument.')


def main() -> None:
    config = RecomposerConfig()
    black_box = RecomposerExperiment(config)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
