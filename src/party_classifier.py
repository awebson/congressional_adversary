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
        self.cherry_pick = config.cherry_pick
        self.device = config.device

        if data.pretrained_embedding is not None:
            config.embed_size = data.pretrained_embedding.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                data.pretrained_embedding)
        else:
            self.embedding = nn.Embedding(
                len(self.word_to_id), config.embed_size)
            init_range = 1.0 / config.embed_size
            nn.init.uniform_(self.embedding.weight.data,
                             -init_range, init_range)
            # nn.init.normal_(self.embedding_bag.weight.data, 0, init_range)
        self.embedding.weight.requires_grad = not config.freeze_embedding

        # Minimal working example when not freezing embeddings
        # self.classifier = nn.Linear(
        #     config.embed_size, config.num_prediction_classes)

        # One extra layer for frozen embeddings
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_size, config.hidden_size),
            # nn.ReLU(),
            # nn.Dropout(p=config.dropout_p),
            nn.Linear(config.hidden_size, config.num_prediction_classes)
        )

        self.to(self.device)

    def forward(self, word_ids: Vector) -> Vector:
        features = self.embedding(word_ids)
        logits = self.classifier(features)
        return logits

    def predict(self, word_ids: Vector) -> Vector:
        self.eval()
        with torch.no_grad():
            logits = self.forward(word_ids)
            confidence = nn.functional.softmax(logits, dim=1)
        self.train()
        return confidence

    def accuracy(
            self,
            batch: Tuple,
            export_error_analysis: Optional[str] = None
            ) -> float:
        word_ids = batch[1].to(self.device)
        labels = batch[0].to(self.device)
        confidence = self.predict(word_ids)
        predictions = confidence.argmax(dim=1)
        correct_indicies = predictions.eq(labels)
        accuracy = correct_indicies.float().mean().item()
        return accuracy

    def all_vocab_confidence(
            self,
            export_path: Optional[str] = None
            ) -> Vector:
        all_vocab_ids = torch.arange(
            len(self.word_to_id), dtype=torch.long, device=self.device)
        confidence = self.predict(all_vocab_ids)
        if not export_path:
            return confidence

        tqdm.write(f'Exporting all vocabulary confidence to {export_path}')
        output = []
        for conf, word_id in zip(confidence, all_vocab_ids):  # type: ignore
            word = self.id_to_word[word_id.item()]
            output.append((conf.tolist(), word))
        output.sort(key=lambda tup: tup[0][0])  # ascending GOP confidence

        if self.cherry_pick:
            cherry_output = []
            for cherry_word in self.cherry_pick:
                cherry_conf = confidence[self.word_to_id[cherry_word]]
                cherry_output.append(
                    (cherry_conf.tolist(), cherry_word))
            cherry_output.sort(key=lambda tup: tup[0][0])

        with open(export_path, 'w') as file:
            file.write('[Dem confidence, GOP confidence]\n')
            if self.cherry_pick:
                for conf, word in cherry_output:
                    file.write(f'[{conf[0]:.2%}, {conf[1]:.2%}]\t{word}\n')
                file.write('\n')
            for conf, word in output:
                file.write(f'[{conf[0]:.2%}, {conf[1]:.2%}]\t{word}\n')

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
        # self.valid_batch = (
        #     torch.tensor([label for label, _ in test]),
        #     torch.tensor([word_id for _, word_id in test])
        # )

        # vocab_size = len(self.word_to_id)
        # self.error_analysis_batch = (
        #     torch.zeros(vocab_size),    # placeholder party labels
        #     torch.arange(vocab_size))   # all word ids

    def __len__(self) -> int:
        return len(self.train_data)

    def __getitem__(self, index: int) -> Tuple[int, int]:
        return self.train_data[index][0], self.train_data[index][1]

    @staticmethod
    def load_embeddings_from_plain_text(path: str) -> Tuple[
            Matrix,
            Dict[str, int],
            Dict[int, str]]:
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

    def __init__(self, config: 'NaiveWordClassifierConfig'):
        super().__init__(config)
        self.data = NaiveWordClassifierDataset(config)
        self.dataloader = DataLoader(
            self.data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_dataloader_threads)
        self.model = NaiveWordClassifier(config, self.data)
        self.optimizer = config.optimizer(
            self.model.parameters(),
            lr=config.learning_rate)
        # self.lr_scheduler = config.lr_scheduler(self.optimizer)
        self.to_be_saved = {
            'config': self.config,
            'model': self.model,
            'optimizer': self.optimizer}

    def train(self) -> None:
        config = self.config
        cross_entropy = nn.CrossEntropyLoss()
        epoch_pbar = tqdm(range(1, config.num_epochs + 1), desc='Epochs')
        for epoch_index in epoch_pbar:
            batch_pbar = tqdm(
                enumerate(self.dataloader), total=len(self.dataloader),
                mininterval=1, desc='Batches')
            for batch_index, batch in batch_pbar:
                party_labels = batch[0].to(config.device)
                word_ids = batch[1].to(config.device)

                self.optimizer.zero_grad()
                logits = self.model(word_ids)
                loss = cross_entropy(logits, party_labels)
                loss.backward()
                self.optimizer.step()

                if batch_index % config.update_tensorboard == 0:
                    stats = {
                        'Loss/train': loss.item(),
                        'Accuracy/train': self.model.accuracy(batch)}
                    self.update_tensorboard(stats)
                if batch_index % config.print_stats == 0:
                    self.print_stats(epoch_index, batch_index, stats)
            # End Batches
            self.print_timestamp()
            # self.lr_scheduler.step()
            self.auto_save(epoch_index)

            if config.export_error_analysis:
                if (epoch_index % config.export_error_analysis == 0
                        or epoch_index == 1):
                    export_path = os.path.join(
                        config.output_dir,
                        f'error_analysis_epoch{epoch_index}.txt')
                    self.model.all_vocab_confidence(export_path)
        # End Epochs


@dataclass
class NaiveWordClassifierConfig():

    # Essential
    input_dir: str = '../data/processed/word_classifier/44_Obama_1e-5'
    output_dir: str = '../results/party_classifier/word/frozen'
    device: torch.device = torch.device('cuda:0')

    # Hyperparameters
    embed_size: int = 300
    hidden_size: int = 100
    batch_size: int = 4000
    num_epochs: int = 10

    pretrained_embedding: Optional[str] = None
    freeze_embedding: bool = True
    dropout_p: float = 0

    optimizer: torch.optim.Optimizer = torch.optim.Adam
    learning_rate: float = 1e-3

    num_prediction_classes: int = 2
    # num_valid_holdout: int = 10_000
    # num_test_holdout: int = 10_000
    num_dataloader_threads: int = 0

    # Evaluation
    cherry_pick: Optional[Tuple[str, ...]] = (
        'estate_tax', 'death_tax',
        'immigrants', 'illegal_immigrants', 'illegal_aliens',
        'protection_and_affordable', 'the_affordable_care_act',
        'obamacare', 'health_care_bill', 'socialized_medicine', 'public_option',
        'the_wall_street_reform_legislation',
        'financial_stability',
        'capital_gains_tax',
        'second_amendment_rights',
        'government_spending',
        'deficit_spending',
        'bush_tax_cuts'
    )

    # Housekeeping
    export_error_analysis: Optional[int] = 1  # per epoch
    update_tensorboard: int = 1_000  # per batch
    print_stats: int = 1_000  # per batch
    reload_path: Optional[str] = None
    clear_tensorboard_log_in_output_dir: bool = True
    delete_all_exisiting_files_in_output_dir: bool = False
    auto_save: bool = False
    auto_save_per_epoch: Optional[int] = None
    save_to_tensorboard_embedding_projector: bool = False


def main() -> None:
    # TODO try sparse gradient & optimizer
    config = NaiveWordClassifierConfig(
        output_dir='../results/party_classifier/word/linear_MLP',
        # pretrained_embedding='../results/baseline/word2vec_Obama.txt',
        freeze_embedding=True,
        batch_size=1000,
        hidden_size=300,
        num_epochs=1000,
        learning_rate=1e-3,
        device=torch.device('cuda:0')
    )

    black_box = NaiveWordClassifierExperiment(config)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    main()
