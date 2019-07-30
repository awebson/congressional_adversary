import functools
import pickle
import random
import os
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from typing import Counter as CounterType

import numpy as np
import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm

from utils.experiment import Experiment, ExperimentConfig
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class SkipGramNegativeSampling(nn.Module):
    """standard skip-gram negative sampling model"""

    def __init__(
            self,
            config: 'SkipGramConfig',
            data: 'SkipGramDataset'
            ) -> None:
        """
        Note the initialization trick.
        """
        super().__init__()
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        vocab_size = len(data.word_to_id)
        embed_size = config.embed_size
        self.num_negative_samples = config.num_negative_samples
        self.device = config.device

        self.center_embedding = nn.Embedding(
            vocab_size, embed_size, sparse=config.sparse_embedding)
        init_range = 1.0 / embed_size
        nn.init.uniform_(self.center_embedding.weight.data, -init_range, init_range)

        self.context_embedding = nn.Embedding(
            vocab_size, embed_size, sparse=config.sparse_embedding)
        nn.init.constant_(self.context_embedding.weight.data, 0)

        self.init_negative_sampling(data.word_frequency, data.word_to_id)
        self.to(self.device)

    def init_negative_sampling(
            self,
            word_frequency: CounterType[str],
            word_to_id: Dict[str, int]
            ) -> None:
        """
        A smoothed unigram distribution.
        A simplified case of Noise Contrastive Estimate.
        where the seemingly arbitrary number of 0.75 is from Mikolov 2014.
        """
        message = (
            'As of PyTorch 1.1, torch.distributions.categorical.Categorical '
            'is very slow, so a third-party alternative is necessary for now: '
            'https://pypi.org/project/pytorch-categorical/'
        )
        try:
            import pytorch_categorical
        except ImportError as error:
            print(message)
            raise error
        cumulative_freq = sum(freq ** 0.75 for freq in word_frequency.values())
        categorical_dist_probs: Dict[int, float] = {
            word_to_id[word]: (freq ** 0.75) / cumulative_freq
            for word, freq in word_frequency.items()
        }
        vocab_size = len(word_to_id)
        categorical_dist_probs: Vector = torch.tensor([
            # categorical_dist_probs[word_id]  # strict
            categorical_dist_probs.get(word_id, 0)  # prob = 0 if missing vocab
            for word_id in range(vocab_size)
        ])
        self.negative_sampling_dist = pytorch_categorical.Categorical(
            categorical_dist_probs, self.device)

    # def forward(
    #         self,
    #         center_ids: Vector,
    #         context_ids: Vector
    #         ) -> Scalar:
    #     center_vectors = self.center_embedding(center_ids)
    #     context_vectors = self.context_embedding(context_ids)

    #     batch_size = len(center_ids)
    #     neg_center_ids = center_ids.unsqueeze(1).expand(
    #         batch_size, self.num_negative_samples)
    #     neg_context_ids = self.negative_sampling_dist.sample(
    #         (batch_size, self.num_negative_samples))
    #     neg_center_vectors = self.center_embedding(neg_center_ids)
    #     neg_context_vectors = self.context_embedding(neg_context_ids)

    #     # batch_size * embed_size
    #     pos_objective = torch.einsum(
    #         'bd,bd->b', center_vectors, context_vectors)
    #     pos_objective = nn.functional.logsigmoid(pos_objective)

    #     # batch_size * num_negative_examples * embed_size
    #     neg_objective = torch.einsum(
    #         'bnd,bnd->b', neg_center_vectors, neg_context_vectors)
    #     neg_objective = nn.functional.logsigmoid(0 - neg_objective)
    #     return 0 - torch.mean(pos_objective + neg_objective)

    def forward(
            self,
            center_ids: Vector,
            context_ids: Vector
            ) -> Scalar:
        """
        Reference implementation
        """
        center_vectors = self.center_embedding(center_ids)
        context_vectors = self.context_embedding(context_ids)

        score = torch.sum(torch.mul(center_vectors, context_vectors), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -nn.functional.logsigmoid(score)

        batch_size = len(center_ids)
        neg_context_ids = self.negative_sampling_dist.sample(
            (batch_size, self.num_negative_samples))
        neg_context_vectors = self.context_embedding(neg_context_ids)

        neg_score = torch.bmm(neg_context_vectors, center_vectors.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(nn.functional.logsigmoid(-neg_score), dim=1)
        return torch.mean(score + neg_score)


class LegacySkipGramDataset(torch.utils.data.Dataset):

    def __init__(self, config: 'SkipGramConfig'):
        self.window_radius = config.window_radius
        vocab_path = os.path.join(config.input_dir, 'vocab.pickle')
        with open(vocab_path, 'rb') as vocab_file:
            preprocessed = pickle.load(vocab_file)
        self.word_to_id = preprocessed[0]
        self.id_to_word = preprocessed[1]
        self.word_frequency: CounterType[str] = preprocessed[2]

        corpus_path = os.path.join(config.input_dir, 'train_data.pickle')
        print(f'Loading corpus at {corpus_path}', flush=True)
        with open(corpus_path, 'rb') as corpus_file:
            tokenized_documents: List[List[int]] = pickle.load(corpus_file)
        if config.debug_subset_corpus:
            tokenized_documents = tokenized_documents[:config.debug_subset_corpus]
        self.tokenized_documents: List[List[int]] = tokenized_documents

    def __len__(self) -> int:
        return len(self.tokenized_documents)

    def __getitem__(self, index: int) -> List[Tuple[int, List[int]]]:
        """
        parse one document into a List[skip-grams], where each skip-gram
        is a Tuple[center_id, List[context_ids]]
        """
        doc: List[int] = self.tokenized_documents[index]
        skip_grams: List[Tuple[int, List[int]]] = []
        for center_index, center_word_id in enumerate(doc):
            left_index = max(center_index - self.window_radius, 0)
            right_index = min(center_index + self.window_radius, len(doc) - 1)
            context_word_ids: List[int] = (
                doc[left_index:center_index] +
                doc[center_index + 1:right_index + 1])
            skip_grams.append((center_word_id, context_word_ids))
        return skip_grams

    @staticmethod
    def collate(
            faux_batch: List[List[Tuple[int, List[int]]]]
            ) -> Tuple[Matrix, R3Tensor]:
        center_vector = []
        context_vector = []
        for list_skip_grams in faux_batch:  # each list is parsed from a doc
            for center_id, context_ids in list_skip_grams:
                center_vector += [center_id for _ in range(len(context_ids))]
                context_vector += context_ids
                # TODO try torch.expand here
        return torch.tensor(center_vector), torch.tensor(context_vector)


class SkipGramDataset(torch.utils.data.Dataset):

    def __init__(self, config: 'SkipGramConfig'):
        self.window_radius = config.window_radius
        vocab_path = os.path.join(config.input_dir, 'vocab.pickle')
        with open(vocab_path, 'rb') as vocab_file:
            preprocessed = pickle.load(vocab_file)
        self.word_to_id = preprocessed[0]
        self.id_to_word = preprocessed[1]
        self.word_frequency: CounterType[str] = preprocessed[2]

        corpus_path = os.path.join(config.input_dir, 'train_data.pickle')
        print(f'Loading corpus at {corpus_path}', flush=True)
        with open(corpus_path, 'rb') as corpus_file:
            tokenized_documents: List[List[int]] = pickle.load(corpus_file)
        if config.debug_subset_corpus:
            tokenized_documents = tokenized_documents[:config.debug_subset_corpus]
        self.tokenized_documents: List[List[int]] = tokenized_documents

    def __len__(self) -> int:
        return len(self.tokenized_documents)

    def __getitem__(self, index: int) -> Tuple[Vector, Vector]:
        """
        parse one document of word_ids into a flatten list of skip-grams
        """
        doc = self.tokenized_documents[index]
        center_word_ids: List[int] = []
        context_word_ids: List[int] = []
        for center_index, center_word_id in enumerate(doc):
            left_index = max(center_index - self.window_radius, 0)
            right_index = min(center_index + self.window_radius, len(doc) - 1)
            context_word_id: List[int] = (
                doc[left_index:center_index] +
                doc[center_index + 1:right_index + 1])
            center_word_ids += [center_word_id] * len(context_word_id)
            context_word_ids += context_word_id
        return torch.tensor(center_word_ids), torch.tensor(context_word_ids)

    @staticmethod
    def collate(
            faux_batch: List[Tuple[Vector, Vector]]
            ) -> Tuple[Vector, Vector]:
        center_word_ids = torch.cat([center for center, _ in faux_batch])
        context_word_ids = torch.cat([context for _, context in faux_batch])
        return center_word_ids, context_word_ids

    # @staticmethod
    # def collate(
    #         faux_batch: List[Tuple[Vector, Vector]]
    #         ) -> Tuple[Vector, Vector]:
    #     x = []
    #     y = []
    #     for st, uff in faux_batch:
    #         x.append(st)
    #         y.append(uff)
    #     center_word_ids = torch.cat(x)
    #     context_word_ids = torch.cat(y)
    #     return center_word_ids, context_word_ids


class SlowSkipGramDataset(SkipGramDataset):

    def __getitem__(self, index: int) -> List[Tuple[int, int]]:
        """
        parse one document of word_ids into a flatten list of skip-grams
        """
        doc = self.tokenized_corpus[index]
        skip_grams: List[Tuple[int, int]] = []
        for center_index, center in enumerate(doc):
            left_index = max(center_index - self.window_radius, 0)
            right_index = min(center_index + self.window_radius, len(doc) - 1)
            # left_index = center_index - self.window_radius
            # right_index = center_index + self.window_radius
            # if left_index < 0 or right_index >= len(doc):
            #     continue
            contexts: List[int] = (
                doc[left_index:center_index] +
                doc[center_index + 1:right_index + 1])
            skip_grams += [(center, context) for context in contexts]
        return skip_grams

    @staticmethod
    def collate(
            faux_batch: List[List[Tuple[int, List[int]]]]
            ) -> Tuple[Matrix, R3Tensor]:
        center_vector = []
        context_vector = []
        for list_skip_grams in faux_batch:  # each list is parsed from a doc
            for center_id, context_ids in list_skip_grams:
                center_vector += [center_id for _ in range(len(context_ids))]
                context_vector += context_ids
                # TODO try torch.expand here
        return torch.tensor(center_vector), torch.tensor(context_vector)


class SkipGramExperiment(Experiment):

    def train(self) -> None:
        config = self.config
        tb_global_step = 0
        epoch_progress = tqdm(range(1, config.num_epochs + 1), desc='Epochs')
        for epoch_index in epoch_progress:
            batch_progress = tqdm(
                enumerate(self.dataloader), total=len(self.dataloader),
                mininterval=1, desc='Batches')
            for batch_index, batch in batch_progress:
                center_ids = batch[0].to(config.device)
                context_ids = batch[1].to(config.device)

                self.model.zero_grad()
                loss = self.model(center_ids, context_ids)
                loss.backward()
                self.optimizer.step()
                # for profiling
                # if (batch_index + 1) % config.debug_subset_corpus == 0:
                #     return
                if batch_index % config.update_tensorboard_per_batch == 0:
                    self.tensorboard.add_scalar(
                        'training loss', loss.item(), tb_global_step)
                    tb_global_step += 1
                if (batch_index % config.print_stats_per_batch) == 0:
                    self.print_stats(loss.item(), epoch_index, batch_index)
            # end batch

            self.lr_scheduler.step()
            if config.auto_save_every_epoch or epoch_index == config.num_epochs:
                self.save_state_dict(epoch_index, tb_global_step)
        # end epoch
        print('\n✅ Training Complete')


@dataclass
class SkipGramConfig(ExperimentConfig):

    # Essential
    input_dir: str = '../data/processed/skip_gram/44_Obama_1e-5'
    output_dir: str = '../results/skip_gram/debug'
    device: torch.device = torch.device('cuda:1')
    num_dataloader_threads: int = 0
    debug_subset_corpus: Optional[int] = None

    # Hyperparameters
    batch_size: int = 8  # each unit is a document worth of skip-grams
    embed_size: int = 300
    window_radius: int = 5  # context_size = 2 * window_radius
    num_negative_samples: int = 10
    num_epochs: int = 10
    sparse_embedding: bool = True  # faster if use with sparse optimizer

    # Optimizer
    optimizer: functools.partial = functools.partial(
        torch.optim.SparseAdam,
        lr=1e-3)
    lr_scheduler: functools.partial = functools.partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5,
        gamma=0.1)
    # optimizer=functools.partial(
    #     torch.optim.SGD,
    #     lr=0.001)
    # lr_scheduler=functools.partial(
    #     torch.optim.lr_scheduler.StepLR,
    #     step_size=2,
    #     gamma=0.25)

    # Housekeeping
    reload_state_dict_path: Optional[str] = None
    reload_experiment_path: Optional[str] = None
    clear_tensorboard_log_in_output_dir: bool = True
    delete_all_exisiting_files_in_output_dir: bool = False
    auto_save_every_epoch: bool = False

    auto_save_before_quit: bool = True
    save_to_tensorboard_embedding_projector: bool = False
    update_tensorboard_per_batch: int = 1_000
    print_stats_per_batch: int = 1_000


def main() -> None:
    config = SkipGramConfig()
    data = SkipGramDataset(config)
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data.collate,
        num_workers=config.num_dataloader_threads)
    model = SkipGramNegativeSampling(config, data)
    optimizer = config.optimizer(model.parameters())
    lr_scheduler = config.lr_scheduler(optimizer)
    black_box = SkipGramExperiment(
        config, data, dataloader, model, optimizer, lr_scheduler)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    main()
