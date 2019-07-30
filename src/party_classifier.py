import functools
import pickle
import random
import os
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.experiment import Experiment, ExperimentConfig
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor

random.seed(42)
torch.manual_seed(42)

class NaiveWordClassifier(nn.Module):

    def __init__(
            self,
            config: 'NaiveWordClassifierConfig',
            data: 'NaiveWordClassifierDataset'):
        # nn.Module.__init__(self)
        super().__init__()
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.device = config.device

        if getattr(data, 'pretrained_embedding') is not None:
            config.embed_size = data.pretrained_embedding.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                data.pretrained_embedding)
        else:
            self.embedding = nn.Embedding(
                len(self.word_to_id), config.embed_size)
            init_range = 1.0 / config.embed_size
            nn.init.uniform_(
                self.embedding.weight.data, -init_range, init_range)
            # nn.init.normal_(self.embedding_bag.weight.data, 0, init_range)
        self.embedding.weight.requires_grad = not config.freeze_embedding

        self.dropout = nn.Dropout(p=config.dropout_p)
        self.linear = nn.Linear(config.embed_size, config.num_prediction_classes)

        # self.MLP_classifier = nn.Sequential(
        #     nn.Linear(config.embed_size, config.embed_size),
        #     nn.ReLU(),
        #     nn.Dropout(p=config.dropout_p),
        #     nn.Linear(config.embed_size, config.num_prediction_classes))

        self.to(self.device)

    def forward(self, word_ids: Vector) -> Vector:
        features = self.embedding(word_ids)
        logits = self.linear(self.dropout(features))
        # logits = self.MLP_classifier(features)
        return logits

    def accuracy(
            self,
            batch: Tuple,
            export_error_analysis: Optional[str] = None
            ) -> float:
        word_ids = batch[1].to(self.device)
        labels = batch[0].to(self.device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(word_ids)
            predictions = logits.argmax(dim=1)
            accuracy = predictions.eq(labels).float().mean().item()

            if export_error_analysis:
                tqdm.write(
                    f'Exporting error analysis to {export_error_analysis}')
                confidence = nn.functional.softmax(logits, dim=1)
                losses = nn.functional.cross_entropy(logits, labels, reduction='none')
                losses = nn.functional.normalize(losses, dim=0)

                output_iter = []
                for conf, loss, label, word_id in zip(
                        confidence, losses, labels, word_ids):
                    word = self.id_to_word[word_id.item()]
                    correctness = (label == torch.argmax(conf)).item()
                    output_iter.append((conf.tolist(), label, loss, correctness, word))
                output_iter.sort(key=lambda tup: tup[0][0])
                self.accuracy_at_confidence_plot(output_iter)
                with open(export_error_analysis, 'w') as file:
                    file.write(f'accuracy = {accuracy:.4f}\n\n')
                    file.write('(Dem confidence, GOP confidence)\n')
                    for conf, label, loss, correct, word in output_iter:
                        file.write(f'({conf[0]:.2%}, {conf[1]:.2%})\t'
                                   #  f'{label}\t'
                                   #  f'{loss:.2%}\t'
                                   f'{word}\n')
        self.train()
        return accuracy

    @staticmethod
    def accuracy_at_confidence_plot(output_iter: List) -> None:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()

        accuracies = []
        bins = 21
        conf_threshold = np.linspace(0, 1, num=bins)
        for i in range(bins):
            x_confident_correctness = [
                correctness
                for conf, _, _, correctness, _ in output_iter
                if conf_threshold[i] < conf[0] < conf_threshold[i + 1]
            ]
            if len(x_confident_correctness) < 10:
                accuracies.append(0)
            else:
                accuracies.append(np.mean(x_confident_correctness))
            # print(conf_threshold[i], x_confident_correctness)
        print(accuracies)
        sns.lineplot(bins, accuracies)
        # plt.legend()
        plt.savefig('x_confident_accuracy.png')

class NaiveWordClassifierDataset(Dataset):

    def __init__(self, config: 'NaiveWordClassifierConfig'):

        train_data_path = os.path.join(config.input_dir, 'train_data.pickle')
        print(f'Loading training data at {train_data_path}', flush=True)
        with open(train_data_path, 'rb') as file:
            preprocessed = pickle.load(file)
        # self.word_frequency: Dict[str, int] = preprocessed[2]
        train: List[Tuple[int, int]] = preprocessed[3]
        test: List[Tuple[int, int]] = preprocessed[4]
        if config.pretrained_embedding is not None:
            self.pretrained_embedding, self.word_to_id, self.id_to_word = (
                self.load_embeddings_from_plain_text(config.pretrained_embedding))
        else:
            self.pretrained_embedding = None
            self.word_to_id = preprocessed[0]
            self.id_to_word = preprocessed[1]

        self.train_data = train

        # test_data = sentences[num_train + test_holdout:]
        self.valid_batch = (
            torch.tensor([label for label, _ in test]),
            torch.tensor([word_id for _, word_id in test])
        )

        vocab_size = len(self.word_to_id)
        self.error_analysis_batch = (
            torch.zeros(vocab_size),    # placeholder party labels
            torch.arange(vocab_size))   # all word ids

    def __len__(self) -> int:
        return len(self.train_data)

    def __getitem__(self, index: int) -> Tuple[int, int]:
        return self.train_data[index][0], self.train_data[index][1]

    @staticmethod
    def load_embeddings_from_plain_text(path: str) -> Tuple[
            Matrix,
            Dict[str, int],
            Dict[int, str]]:
        """TODO turn off auto_caching"""
        id_generator = 0
        word_to_id: Dict[str, int] = {}
        id_to_word: Dict[int, str] = {}
        embeddings: List[List[float]] = []  # cast to float when instantiating tensor
        print(f'Loading pretrained embeddings from {path}', flush=True)
        with open(path) as file:
            vocab_size, embed_size = map(int, file.readline().split())
            print(f'vocab_size = {vocab_size:,}, embed_size = {embed_size}')
            for raw_line in file:
                line: List[str] = raw_line.split()
                word = line[0]
                # vector = torch.Tensor(line[-embed_size:], dtype=torch.float32)
                embeddings.append(list(map(float, line[-embed_size:])))
                word_to_id[word] = id_generator
                id_to_word[id_generator] = word
                id_generator += 1
        embeddings = torch.tensor(embeddings)
        return embeddings, word_to_id, id_to_word


class NaiveWordClassifierExperiment(Experiment):

    def train(self) -> None:
        config = self.config
        cross_entropy = nn.CrossEntropyLoss()
        tb_global_step = 0
        epoch_progress = tqdm(range(1, config.num_epochs + 1), desc='Epochs')
        for epoch_index in epoch_progress:
            batch_progress = tqdm(
                enumerate(self.dataloader), total=len(self.dataloader),
                mininterval=1, desc='Batches')
            for batch_index, batch in batch_progress:

                party_labels = batch[0].to(config.device)
                word_ids = batch[1].to(config.device)

                self.optimizer.zero_grad()
                logits = self.model(word_ids)
                loss = cross_entropy(logits, party_labels)
                loss.backward()
                self.optimizer.step()

                if batch_index % config.update_tensorboard_per_batch == 0:
                    batch_accuracy = self.model.accuracy(batch)
                    tqdm.write(f'Epoch {epoch_index}, Batch {batch_index:,}:\t'
                               f'Loss = {loss.item():.5f}\t'
                               f'Batch accuracy = {batch_accuracy:.2%}')

                    self.tensorboard.add_scalar(
                        'training loss', loss.item(), tb_global_step)
                    self.tensorboard.add_scalar(
                        'Batch Accuracy', batch_accuracy, tb_global_step)
                    tb_global_step += 1

                # if (batch_index % config.print_stats_per_batch) == 0:
                #     self.print_stats(loss.item(), epoch_index, batch_index)
            # end batch
            self.lr_scheduler.step()

            valid_accuracy = self.model.accuracy(self.data.valid_batch)
            tqdm.write(f'Epoch {epoch_index}  '
                       f'Validation Accuracy = {valid_accuracy:.2%}\n\n')
            self.tensorboard.add_scalar(
                'validation accuracy', valid_accuracy, epoch_index)
            if config.export_error_analysis:
                if epoch_index == 1 or epoch_index % 10 == 0:
                    valid_accuracy = self.model.accuracy(
                        self.data.valid_batch,  # error_analysis_batch,  # NOTE self.data.valid_batch
                        export_error_analysis=os.path.join(
                            config.output_dir,
                            f'error_analysis_epoch{epoch_index}.txt'))

            if config.auto_save_every_epoch or epoch_index == config.num_epochs:
                self.save_state_dict(epoch_index, tb_global_step)
        # end epoch
        # test_accuracy = self.model.accuracy(self.data.test_batch)
        # self.tensorboard.add_scalar('test accuracy', test_accuracy, global_step=0)
        # print(f'Test Accuracy = {test_accuracy:.2%}')
        print('\n✅ Training Complete')


@dataclass
class NaiveWordClassifierConfig(ExperimentConfig):

    # Essential
    input_dir: str = '../data/processed/word_classifier/UCSB_1e-5'
    output_dir: str = '../results/party_classifier/word/debug'
    device: torch.device = torch.device('cuda:0')

    # Hyperparameters
    embed_size: int = 300
    batch_size: int = 4000
    num_epochs: int = 10

    pretrained_embedding: Optional[str] = None
    # pretrained_embedding: Optional[str] = '../results/baseline/word2vec_president.txt'
    freeze_embedding: bool = False
    dropout_p: float = 0

    # Optimizer
    optimizer: functools.partial = functools.partial(
        torch.optim.Adam,
        lr=1e-3,
        # weight_decay=1e-3
    )
    # optimizer=functools.partial(
    #     torch.optim.SGD,
    #     lr=1e-3,
    #     momentum=0.9),
    lr_scheduler: functools.partial = functools.partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=30,
        gamma=0.1
    )

    num_prediction_classes: int = 2
    num_valid_holdout: int = 10_000
    num_test_holdout: int = 10_000
    num_dataloader_threads: int = 0

    reload_state_dict_path: Optional[str] = None
    reload_experiment_path: Optional[str] = None
    clear_tensorboard_log_in_output_dir: bool = True
    delete_all_exisiting_files_in_output_dir: bool = False
    auto_save_every_epoch: bool = False
    export_error_analysis: bool = False

    auto_save_before_quit: bool = True
    save_to_tensorboard_embedding_projector: bool = False
    update_tensorboard_per_batch: int = 1_000
    print_stats_per_batch: int = 1_000


def main() -> None:
    config = NaiveWordClassifierConfig()
    data = NaiveWordClassifierDataset(config)
    dataloader = DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_dataloader_threads)
    model = NaiveWordClassifier(config, data)
    optimizer = config.optimizer(model.parameters())
    lr_scheduler = config.lr_scheduler(optimizer)

    black_box = NaiveWordClassifierExperiment(
        config, data, dataloader, model, optimizer, lr_scheduler)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    main()
