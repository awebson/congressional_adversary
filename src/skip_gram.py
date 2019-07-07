import functools
import pickle
import random
import os
from typing import Tuple, List, Dict
from typing import Counter as CounterType

import numpy as np
import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm

from utils.experiment import Experiment, ExperimentConfig
# from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class SkipGramNegativeSampling(nn.Module):

    """standard skip-gram negative sampling model"""

    def __init__(
            self,
            word_to_id: Dict[str, int],
            id_to_word: Dict[int, str],
            word_frequency: CounterType[str],
            embed_dim: int,
            window_radius: int,
            num_negative_samples: int,
            device: torch.device
            ) -> None:
        """
        Note the initialization trick.
        """
        super().__init__()
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.vocab_size = len(word_to_id)
        self.embed_dim = embed_dim
        self.context_size = window_radius * 2
        self.num_negative_samples = num_negative_samples
        self.device = device

        self.center_embedding = nn.Embedding(
            self.vocab_size, embed_dim, sparse=True).to(device)
        self.context_embedding = nn.Embedding(
            self.vocab_size, embed_dim, sparse=True).to(device)

        init_range = 1.0 / embed_dim
        nn.init.uniform_(self.center_embedding.weight.data, -init_range, init_range)
        nn.init.constant_(self.context_embedding.weight.data, 0)

        # self.init_negative_sampling(frequency, id_to_word)
        import pytorch_categorical
        self.negative_sampling_dist = pytorch_categorical.Categorical(
            word_frequency, self.device)


    def init_negative_sampling(
            self,
            word_frequency: CounterType[str],
            id_to_word: Dict[int, str]
            ) -> None:
        """
        A smoothed unigram distribution.
        A simplified case of Noise Contrasitive Estimate.
        where the seemingly aribitrary number of 0.75 is from the paper.
        """
        message = (
            'As of PyTorch 1.1, torch.distributions.categorical.Categorical '
            'is very slow, so a thrid-party alternative is necessary for now: '
            'https://pypi.org/project/pytorch-categorical/'
        )
        try:
            import pytorch_categorical
        except ImportError as error:
            print(message)
            raise error
        cumulative_freq = sum(freq ** 0.75 for freq in word_frequency.values())
        sampling_dist: Dict[str, float] = {
            word: (freq ** 0.75) / cumulative_freq
            for word, freq in word_frequency.items()
        }
        categorical_dist_probs = [
            sampling_dist.get(id_to_word[word_id], 0)
            for word_id in range(len(id_to_word))
        ]
        self.negative_sampling_dist = pytorch_categorical.Categorical(
            categorical_dist_probs, self.device)

    # def forward(
    #         self,
    #         center_ids: Vector,   # batch_size
    #         context_ids: Vector  # batch_size
    #         ) -> Vector:
    #     """
    #     Here in shape comments, noise_size is synonymous with num_negative_examples.

    #     Noted by Mikolov 2014, negative sampling is a simplified version of
    #     noise contrasitve estimation (NCE).
    #     """
    #     center_vectors = self.center_embedding(center_ids)
    #     context_vectors = self.context_embedding(context_ids)

    #     # batch_size * embed_dim
    #     pos_objective = torch.einsum('bd,bd->b', center_vectors, context_vectors)
    #     pos_objective = nn.functional.logsigmoid(pos_objective)

    #     batch_size = len(center_ids)
    #     neg_center_ids = center_ids.unsqueeze(1).expand(
    #         batch_size, self.num_negative_samples)
    #     neg_context_ids = self.negative_sampling_dist.sample(
    #         (batch_size, self.num_negative_samples))
    #     neg_center_vectors = self.center_embedding(neg_center_ids)
    #     neg_context_vectors = self.context_embedding(neg_context_ids)
    #     # batch_size * num_negative_examples * embed_dim
    #     neg_objective = torch.einsum('bnd,bnd->b', neg_center_vectors, neg_context_vectors)
    #     neg_objective = nn.functional.logsigmoid(0 - neg_objective)

    #     return 0 - torch.mean(pos_objective + neg_objective)

    def forward(self, batch):
        """
        Reference implementation
        """
        center_ids = batch[0].to(self.device)
        context_ids = batch[1].to(self.device)

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


class TrueSkipGramDataset(torch.utils.data.Dataset):

    def __init__(self):
        vocab_path = os.path.join(args.corpus_dir, 'vocab.pickle')
        with open(vocab_path, 'rb') as vocab_file:
            preprocessed = pickle.load(vocab_file)
        self.word_to_id = preprocessed[0]
        self.id_to_word = preprocessed[1]
        self.negative_sampling_dist = torch.Tensor(preprocessed[2])

        corpus_path = os.path.join(args.corpus_dir, 'train_data_0.pickle')
        print(f'Loading corpus at {corpus_path}', flush=True)
        with open(corpus_path, 'rb') as corpus_file:
            self.tokenized_corpus: List[List[int]] = pickle.load(corpus_file)

        print('Indexing corpus...', flush=True)
        mapping: Dict[int, Tuple[int, int]] = {}
        master_index = 0
        self.master_num_words = 0
        for doc_index, doc in enumerate(self.tokenized_corpus):
            self.master_num_words += len(doc)
            for center_word_index in range(len(doc)):
                mapping[master_index] = (doc_index, center_word_index)
                master_index += 1
        self.mapping = mapping

    def __len__(self):
        return self.master_num_words

    def __getitem__(self, master_index: int) -> Tuple[int, List[int]]:
        """
        Given any aribtirary word anywhere in the corpus,
        parse that center word into a list of skip-grams.
        """
        doc_index, center_index = self.mapping[master_index]
        doc = self.tokenized_corpus[doc_index]
        left_index = max(center_index - args.window_radius, 0)
        right_index = min(center_index + args.window_radius, len(doc) - 1)
        center_word_id = doc[center_index]
        context_word_ids: List[int] = (
            doc[left_index:center_index] +
            doc[center_index + 1:right_index + 1])
        return center_word_id, context_word_ids

    @staticmethod
    def collate(batch):
        """
        flatten batches is necessary because not all words have
        the same context size, while tensors require homogenous shapes
        """
        center_vector = []
        context_vector = []
        for center_id, context_ids in batch:
            center_vector += [center_id for _ in range(len(context_ids))]
            context_vector += context_ids
        return torch.LongTensor(center_vector), torch.LongTensor(context_vector)


class SkipGramDataset(torch.utils.data.Dataset):

    def __init__(self, corpus_dir: str, window_radius: int):
        self.window_radius = window_radius
        vocab_path = os.path.join(corpus_dir, 'vocab.pickle')
        with open(vocab_path, 'rb') as vocab_file:
            preprocessed = pickle.load(vocab_file)
        self.word_to_id = preprocessed[0]
        self.id_to_word = preprocessed[1]
        self.negative_sampling_dist = torch.Tensor(preprocessed[2])

        corpus_path = os.path.join(corpus_dir, 'train_data_0.pickle')
        print(f'Loading corpus at {corpus_path}', flush=True)
        with open(corpus_path, 'rb') as corpus_file:
            tokenized_corpus: List[List[int]] = pickle.load(corpus_file)
        self.tokenized_corpus = [doc for doc in tokenized_corpus if len(doc) > 1]

    def __len__(self):
        return len(self.tokenized_corpus)

    def __getitem__(self, index: int) -> List[Tuple[int, List[int]]]:
        """
        parse one document into a clever list of skip-grams
        """
        doc = self.tokenized_corpus[index]
        skip_grams: List[Tuple[int, List[int]]] = []
        for center_index, center_word_id in enumerate(doc):
            left_index = max(center_index - self.window_radius, 0)
            right_index = min(center_index + self.window_radius, len(doc) - 1)
            context_word_ids: List[int] = (
                doc[left_index:center_index] +
                doc[center_index + 1:right_index + 1])
            skip_grams.append((center_word_id, context_word_ids))
        return skip_grams

    # def __getitem__(self, index: int) -> List[Tuple[int, int]]:
    #     """
    #     parse one document of word_ids into a flatten list of skip-grams
    #     """
    #     doc = self.tokenized_corpus[index]
    #     skip_grams: List[Tuple[int, int]] = []
    #     for center_index, center in enumerate(doc):
    #         left_index = max(center_index - args.window_radius, 0)
    #         right_index = min(center_index + args.window_radius, len(doc) - 1)
    #         # left_index = center_index - args.window_radius
    #         # right_index = center_index + args.window_radius
    #         # if left_index < 0 or right_index >= len(doc):
    #         #     continue
    #         contexts: List[int] = (
    #             doc[left_index:center_index] +
    #             doc[center_index + 1:right_index + 1])
    #         skip_grams += [(center, context) for context in contexts]
    #     return skip_grams

    @staticmethod
    def collate(faux_batch):
        center_vector = []
        context_vector = []
        for list_skip_grams in faux_batch:  # each list is parsed from a doc
            for center_id, context_ids in list_skip_grams:
                center_vector += [center_id for _ in range(len(context_ids))]
                context_vector += context_ids
                # TODO try torch.expand here
        return torch.LongTensor(center_vector), torch.LongTensor(context_vector)


class SkipGramConfig(ExperimentConfig):

    num_embedding_dimension: int
    window_radius: int  # context_size = 2 * window_radius
    num_negative_samples: int
    num_dataloader_threads: int = 0
    # debug_subset_corpus: Optional[int] = None

    num_embedding_dimension: int = 300  #TODO
    window_radius: int = 5  # context_size = 2 * window_radius
    num_negative_samples: int = 10
    num_dataloader_threads: int = 0


class SkipGramExperiment(Experiment):

    def __init__(self, config: SkipGramConfig):
        super().__init__(config)
        self.data = SkipGramDataset(config.corpus_dir, config.window_radius)
        self.dataloader = torch.utils.data.DataLoader(
            self.data,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.data.collate,
            num_workers=config.num_dataloader_threads)
        self.model = SkipGramNegativeSampling(
            self.data.word_to_id,
            self.data.id_to_word,
            self.data.negative_sampling_dist,
            config.num_embedding_dimension,
            config.window_radius,
            config.num_negative_samples,
            config.device)
        self.optimizer = config.optimizer(self.model.parameters())
        self.lr_scheduler = config.lr_scheduler(self.optimizer)

    def train(self) -> None:
        config = self.config
        tb_global_step = 0
        epoch_progress = tqdm(range(1, config.num_epochs + 1), desc='Epochs')
        for epoch_index in epoch_progress:
            batch_progress = tqdm(
                enumerate(self.dataloader), total=len(self.dataloader),
                mininterval=1, desc='Batches')
            for batch_index, batch in batch_progress:
                self.optimizer.zero_grad()
                loss = self.model(batch)
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


def main() -> None:
    config = ExperimentConfig(
        # Essential
        corpus_dir='../data/processed/legacy_skip_gram/decade_paper1e-5',
        output_dir='../results/skip_gram/debug',
        device=torch.device('cuda:0'),
        clear_tensorboard_log_in_output_dir=True,
        delete_all_exisiting_files_in_output_dir=False,
        auto_save_every_epoch=False,

        # Hyperparmeters
        batch_size=32,
        num_embedding_dimension=300,
        window_radius=5,  # context_size = 2 * window_radius
        num_negative_samples=10,
        num_epochs=10,

        # Optimizer
        optimizer=functools.partial(
            torch.optim.SparseAdam,
            lr=1e-3),
        lr_scheduler=functools.partial(
            torch.optim.lr_scheduler.StepLR,
            step_size=5,
            gamma=0.1),
        # optimizer=functools.partial(
        #     torch.optim.SGD,
        #     lr=0.001),
        # lr_scheduler=functools.partial(
        #     torch.optim.lr_scheduler.StepLR,
        #     step_size=2,
        #     gamma=0.25),

        # Housekeeping
        auto_save_before_quit=True,
        save_to_tensorboard_embedding_projector=False,
        print_stats_per_batch=10_000
    )
    with SkipGramExperiment(config) as black_box:
        black_box.train()


if __name__ == '__main__':
    main()
