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

    def __init__(
            self,
            config: 'SkipGramConfig',
            data: 'SkipGramDataset'
            ) -> None:
        super().__init__()
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        vocab_size = len(data.word_to_id)
        embed_size = config.embed_size
        self.num_negative_samples = config.num_negative_samples
        self.device = config.device

        if config.pretrained_embedding is not None:
            config.embed_size = data.pretrained_embedding.shape[1]
            self.center_embedding = nn.Embedding.from_pretrained(
                data.pretrained_embedding, sparse=config.sparse_embedding_grad)
            self.context_embedding = nn.Embedding.from_pretrained(
                data.pretrained_embedding, sparse=config.sparse_embedding_grad)
        else:
            self.center_embedding = nn.Embedding(
                vocab_size, embed_size, sparse=config.sparse_embedding_grad)
            init_range = 1.0 / embed_size
            nn.init.uniform_(self.center_embedding.weight.data,
                             -init_range, init_range)

            self.context_embedding = nn.Embedding(
                vocab_size, embed_size, sparse=config.sparse_embedding_grad)
            nn.init.zeros_(self.context_embedding.weight.data)

        self.center_embedding.weight.requires_grad = not config.freeze_embedding
        self.context_embedding.weight.requires_grad = not config.freeze_embedding

        # Learnability Check
        # self.center_decoder = nn.Sequential(
        #     nn.Linear(config.embed_size, config.embed_size, bias=False),
        #     # nn.PReLU(init=1)
        #     # nn.SELU()
        #     nn.Tanh()
        # )
        # self.context_decoder = nn.Sequential(
        #     nn.Linear(config.embed_size, config.embed_size, bias=False),
        #     # nn.PReLU(init=1)
        #     # nn.SELU()
        #     nn.Tanh()
        # )

        # Self-Normalizing Network
        self.center_decoder = nn.Sequential(
            nn.Linear(config.embed_size, config.embed_size),
            nn.SELU(),
            # nn.AlphaDropout(p=0.2),
            # nn.Linear(config.embed_size, config.embed_size),
            # nn.SELU(),
            # nn.Linear(config.embed_size, config.embed_size),
            # nn.SELU(),
            # nn.Linear(config.embed_size, config.embed_size),
            # nn.SELU(),
        )
        self.context_decoder = nn.Sequential(
            nn.Linear(config.embed_size, config.embed_size),
            nn.SELU(),
            # nn.AlphaDropout(p=0.2),
            # nn.Linear(config.embed_size, config.embed_size),
            # nn.SELU(),
            # nn.Linear(config.embed_size, config.embed_size),
            # nn.SELU(),
            # nn.Linear(config.embed_size, config.embed_size),
            # nn.SELU(),
        )
        nn.init.eye_(self.center_decoder[0].weight)
        nn.init.eye_(self.context_decoder[0].weight)
        nn.init.zeros_(self.center_decoder[0].bias)
        nn.init.zeros_(self.context_decoder[0].bias)

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
        # debug missing vocab
        # for word, freq in word_frequency.items():
        #     if word not in word_to_id:
        #         print(freq, word)

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
    #     """ Readable einsum version"""
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
        https://github.com/Andras7/word2vec-pytorch
        """
        center_vectors = self.center_embedding(center_ids)
        context_vectors = self.context_embedding(context_ids)

        batch_size = len(center_ids)
        neg_context_ids = self.negative_sampling_dist.sample(
            (batch_size, self.num_negative_samples))
        neg_context_vectors = self.context_embedding(neg_context_ids)

        # Learnability Check
        center_vectors = self.center_decoder(center_vectors)
        context_vectors = self.context_decoder(context_vectors)
        neg_context_vectors = self.context_decoder(neg_context_vectors)
        # with torch.no_grad():  # HACK to prevent weird cuda runtime error
        #     _ = torch.matmul(neg_context_vectors[0], center_vectors.unsqueeze(2)[0])

        score = torch.sum(torch.mul(center_vectors, context_vectors), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -nn.functional.logsigmoid(score)

        neg_score = torch.bmm(
            neg_context_vectors, center_vectors.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(nn.functional.logsigmoid(-neg_score), dim=1)
        return torch.mean(score + neg_score)

    # def forward(
    #         self,
    #         center_ids: Matrix,
    #         context_ids: Matrix
    #         ) -> Scalar:
    #     """Faster but less readable."""
    #     center = self.center_embedding(center_ids)
    #     true_context = self.context_embedding(context_ids)

    #     batch_size = len(center_ids)
    #     neg_context_ids = self.negative_sampling_dist.sample(
    #         (batch_size, self.num_negative_samples))
    #     negative_context = self.context_embedding(neg_context_ids)

    #     # batch_size * embed_size
    #     objective = torch.sum(
    #         torch.mul(center, true_context),  # Hadamard product
    #         dim=1)  # be -> b
    #     objective = nn.functional.logsigmoid(objective)

    #     # batch_size * num_negative_examples * embed_size
    #     # negative_context: bne
    #     # center: be -> be1
    #     negative_objective = torch.bmm(  # bne, be1 -> bn1
    #         negative_context, center.unsqueeze(2)
    #         ).squeeze()  # bn1 -> bn
    #     negative_objective = nn.functional.logsigmoid(-negative_objective)
    #     negative_objective = torch.sum(negative_objective, dim=1)  # bn -> b
    #     return -torch.mean(objective + negative_objective)


class SkipGramSoftmax(nn.Module):

    def __init__(
            self,
            config: 'SkipGramConfig',
            data: 'SkipGramDataset'
            ) -> None:
        super().__init__()
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        vocab_size = len(data.word_to_id)
        embed_size = config.embed_size
        self.device = config.device

        if config.pretrained_embedding is not None:
            config.embed_size = data.pretrained_embedding.shape[1]
            self.center_embedding = nn.Embedding.from_pretrained(
                data.pretrained_embedding, sparse=config.sparse_embedding_grad)
        else:
            self.center_embedding = nn.Embedding(
                vocab_size, embed_size, sparse=config.sparse_embedding_grad)
            init_range = 1.0 / embed_size
            nn.init.uniform_(self.center_embedding.weight.data,
                             -init_range, init_range)

        self.center_embedding.weight.requires_grad = not config.freeze_embedding

        self.encoder = nn.Sequential(
            nn.Linear(config.embed_size, config.embed_size),
            nn.SELU(),
            # nn.Linear(config.embed_size, config.embed_size),
            # nn.SELU()
            # nn.ReLU()
        )
        nn.init.eye_(self.encoder[0].weight)
        nn.init.zeros_(self.encoder[0].bias)
        # nn.init.eye_(self.encoder[2].weight)
        # nn.init.zeros_(self.encoder[2].bias)

        self.context_decoder = nn.Linear(embed_size, vocab_size)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.to(self.device)

    def forward(
            self,
            center_ids: Vector,
            context_ids: Vector
            ) -> Scalar:
        encoded = self.encoder(self.center_embedding(center_ids))
        prediction = self.context_decoder(encoded)
        loss = self.cross_entropy(prediction, context_ids)
        return loss


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

        self.pretrained_embedding: Optional[Matrix]
        if config.pretrained_embedding is not None:
            self.pretrained_embedding, self.word_to_id, self.id_to_word = (
                self.load_pretrained_embedding(config.pretrained_embedding))
        else:
            self.pretrained_embedding = None
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

    @staticmethod
    def load_pretrained_embedding(in_path: str) -> Tuple[
            torch.Tensor,
            Dict[str, int],
            Dict[int, str]]:
        id_generator = 0
        word_to_id: Dict[str, int] = {}
        id_to_word: Dict[int, str] = {}
        embeddings: List[List[float]] = []  # cast to float when instantiating tensor
        print(f'Loading pretrained embedding from {in_path}', flush=True)
        with open(in_path) as file:
            vocab_size, embed_size = map(int, file.readline().split())
            print(f'vocab_size = {vocab_size:,}, embed_size = {embed_size}')
            for raw_line in file:
                line: List[str] = raw_line.split()
                word = line[0]
                embeddings.append(list(map(float, line[-embed_size:])))
                word_to_id[word] = id_generator
                id_to_word[id_generator] = word
                id_generator += 1
        embeddings = torch.tensor(embeddings)
        return embeddings, word_to_id, id_to_word


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


class PreparsedSkipGramDataset(SkipGramDataset):

    def __init__(self, config: 'SkipGramConfig'):
        data_path = os.path.join(config.input_dir, 'train_data.pickle')
        with open(data_path, 'rb') as data_file:
            preprocessed = pickle.load(data_file)
        self.word_to_id = preprocessed[0]
        self.id_to_word = preprocessed[1]
        self.word_frequency: CounterType[str] = preprocessed[2]
        self.center_ids: List[int] = preprocessed[3]
        self.context_ids: List[int] = preprocessed[4]

        self.pretrained_embedding: Optional[Matrix]
        if config.pretrained_embedding is not None:
            corpus_id_to_word = self.id_to_word
            self.pretrained_embedding, self.word_to_id, self.id_to_word = (
                Experiment.load_embedding(config.pretrained_embedding))
            self.center_ids = Experiment.convert_word_ids(
                self.center_ids, corpus_id_to_word, self.word_to_id)
            self.context_ids = Experiment.convert_word_ids(
                self.context_ids, corpus_id_to_word, self.word_to_id)

    def __len__(self) -> int:
        return len(self.center_ids)

    def __getitem__(self, index: int) -> Tuple[int, int]:
        return self.center_ids[index], self.context_ids[index]


class SkipGramExperiment(Experiment):

    def __init__(self, config: 'SkipGramConfig'):
        super().__init__(config)
        data = PreparsedSkipGramDataset(config)
        self.dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=config.batch_size,
            shuffle=True,
            # collate_fn=data.collate,
            num_workers=config.num_dataloader_threads)
        self.model = SkipGramNegativeSampling(config, data)
        self.optimizer = config.optimizer(
            self.model.parameters(),
            lr=config.learning_rate)
        # self.lr_scheduler = config.lr_scheduler(optimizer)
        self.to_be_saved = {
            'config': self.config,
            'model': self.model,
            'optimizer': self.optimizer
        }

    def train(self) -> None:
        config = self.config
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
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                if batch_index % config.update_tensorboard == 0:
                    stats = {'Loss/train': loss.item()}
                    self.update_tensorboard(stats)
                if batch_index % config.print_stats == 0:
                    self.print_stats(epoch_index, batch_index, stats)
            # End Batches
            # self.lr_scheduler.step()
            self.print_timestamp()
            self.auto_save(epoch_index)
        # End Epochs


@dataclass
class SkipGramConfig():

    # Essential
    input_dir: str = '../data/processed/skip_gram/Obama_1e-5'
    output_dir: str = '../results/skip_gram/debug'
    device: torch.device = torch.device('cuda')

    # Hyperparameters
    batch_size: int = 8  # each unit is a document worth of skip-grams
    embed_size: int = 300
    window_radius: int = 5  # context_size = 2 * window_radius
    num_negative_samples: int = 10
    num_epochs: int = 10
    freeze_embedding: bool = False
    sparse_embedding_grad: bool = False  # faster with sparse optimizer
    pretrained_embedding: Optional[str] = None

    # Optimizer
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    # optimizer: torch.optim.Optimizer = torch.optim.SparseAdam
    learning_rate: float = 1e-3
    # lr_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.StepLR
    # lr_scheduler_step_size: int = 5
    # lr_scheduler_gamma: float = 0.1

    # Housekeeping
    print_stats: int = 1_000
    update_tensorboard: int = 1_000
    auto_save_per_epoch: Optional[int] = 10
    auto_save_if_interrupted: bool = False
    clear_tensorboard_log_in_output_dir: bool = True
    delete_all_exisiting_files_in_output_dir: bool = False
    export_tensorboard_embedding_projector: bool = False
    num_dataloader_threads: int = 0


def main() -> None:
    # config = SkipGramConfig(
    #     input_dir='../data/processed/skip_gram_preparsed/Obama_1e-5',
    #     output_dir='../results/skip_gram/softmax/frozen w2v ReLU',
    #     device=torch.device('cuda:0'),
    #     pretrained_embedding='../results/baseline/word2vec_Obama.txt',
    #     freeze_embedding=True,
    #     learning_rate=1e-3,
    #     batch_size=512,
    #     print_stats=10_000,
    #     num_epochs=30,
    #     auto_save_per_epoch=3,
    #     # num_dataloader_threads=8
    # )

    # # config = SkipGramConfig(
    # #     input_dir='../data/processed/skip_gram_preparsed/Obama_1e-5',
    # #     output_dir='../results/skip_gram/softmax/frozen w2v ReLU NoInitTrick HBS',
    # #     device=torch.device('cuda:1'),
    # #     pretrained_embedding='../results/baseline/word2vec_Obama.txt',
    # #     freeze_embedding=True,
    # #     learning_rate=1e-3,
    # #     batch_size=1024,
    # #     print_stats=10_000,
    # #     num_epochs=30,
    # #     auto_save_per_epoch=3,
    # # )

    config = SkipGramConfig(
        input_dir='../data/processed/skip_gram_preparsed/Obama_1e-5',
        output_dir='../results/skip_gram/fixedSGNS/SELU BS128',
        device=torch.device('cuda:0'),
        pretrained_embedding='../results/baseline/word2vec_Obama.txt',
        freeze_embedding=True,
        learning_rate=1e-4,
        batch_size=128,
        print_stats=10_000,
        num_epochs=30,
        auto_save_per_epoch=3,
        # num_dataloader_threads=8
    )

    # config = SkipGramConfig(
    #     input_dir='../data/processed/skip_gram_preparsed/Obama_1e-5',
    #     output_dir='../results/skip_gram/fixedSGNS/SELU BS4096',
    #     device=torch.device('cuda:1'),
    #     pretrained_embedding='../results/baseline/word2vec_Obama.txt',
    #     freeze_embedding=True,
    #     learning_rate=1e-4,
    #     batch_size=4096,
    #     print_stats=10_000,
    #     num_epochs=30,
    #     auto_save_per_epoch=3,
    # )


    black_box = SkipGramExperiment(config)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    main()
