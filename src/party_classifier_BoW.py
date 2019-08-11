import functools
import pickle
import random
import os
from dataclasses import dataclass
from typing import Tuple, NamedTuple, List, Dict, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.experiment import Experiment, ExperimentConfig
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor
from preprocessing.S4_export_training_corpus import Sentence
# class Sentence(NamedTuple):
#     word_ids: List[int]
#     party: int  # Dem -> 0; GOP -> 1.

random.seed(42)
torch.manual_seed(42)

class BoWClassifier(nn.Module):

    def __init__(
            self,
            config: 'BoWClassifierConfig',
            data: 'BoWClassifierDataset'):
        nn.Module.__init__(self)
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.device = config.device

        if getattr(data, 'pretrained_embedding') is not None:
            config.embed_size = data.pretrained_embedding.shape[1]
            self.embedding_bag = nn.EmbeddingBag.from_pretrained(
                data.pretrained_embedding, mode='mean')
        else:
            self.embedding_bag = nn.EmbeddingBag(
                len(self.word_to_id), config.embed_size, mode='mean')
            init_range = 1.0 / config.embed_size
            nn.init.uniform_(
                self.embedding_bag.weight.data, -init_range, init_range)
            # nn.init.normal_(self.embedding_bag.weight.data, 0, init_range)
        self.embedding_bag.weight.requires_grad = not config.freeze_embedding

        # self.embedding_bag = nn.EmbeddingBag(
        #     len(self.word_to_id), config.embed_size, mode='mean')
        # init_range = 1.0 / config.embed_size
        # nn.init.uniform_(
        #     self.embedding_bag.weight.data, -init_range, init_range)

        self.dropout = nn.Dropout(p=config.dropout_p)
        self.linear = nn.Linear(config.embed_size, config.num_prediction_classes)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.to(self.device)

    def forward(
            self,
            flatten_sentences: Vector,
            sent_starting_indicies: Vector
            ) -> Vector:
        BoW_embed = self.embedding_bag(flatten_sentences, sent_starting_indicies)
        logits = self.linear(self.dropout(BoW_embed))
        return logits

    # def predict(
    #         self,
    #         flatten_sentences: Vector,
    #         sent_starting_indicies: Vector,
    #         party_labels: Vector
    #         ) -> float:  # accuracy
    #     """always use CPU for evaluation"""
    #     self.eval()
    #     with torch.no_grad():
    #         sent_repr = self.embedding_bag(flatten_sentences, sent_starting_indicies)
    #         logits = self.linear(sent_repr)
    #         predictions = torch.argmax(logits, dim=1)
    #         accuracy = torch.mean(
    #             torch.where(predictions == party_labels,
    #                         torch.ones_like(predictions, dtype=torch.float32),
    #                         torch.zeros_like(predictions, dtype=torch.float32)))
    #     self.train()
    #     return accuracy.item()

    # def eval_validation(self, valid_data: Tuple) -> float:
    #     flatten_sentences = valid_data[0].to(self.device)
    #     sent_starting_indicies = valid_data[1].to(self.device)
    #     party_labels = valid_data[2].to(self.device)
    #     return self.predict(flatten_sentences, sent_starting_indicies, party_labels)

    # def evaluate(self, word_ids: List[int]) -> Vector:
    #     self.eval()
    #     with torch.no_grad():
    #         word_ids = torch.LongTensor(word_ids)
    #         embeddings = self.embedding.cpu()(word_ids)
    #         logits = self.linear.cpu()(embeddings)
    #         confidence = nn.functional.softmax(logits, dim=1)
    #         return confidence

    def accuracy(
            self,
            batch: Tuple,
            export_error_analysis: Optional[str] = None
            ) -> float:
        flatten_sentences = batch[0].to(self.device)
        sent_starting_indicies = batch[1].to(self.device)
        labels = batch[2].to(self.device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(flatten_sentences, sent_starting_indicies)
            predictions = torch.argmax(logits, dim=1)
            accuracy = torch.mean(
                torch.where(predictions == labels,
                            torch.ones_like(predictions, dtype=torch.float32),
                            torch.zeros_like(predictions, dtype=torch.float32)))

            if export_error_analysis:
                tqdm.write(
                    f'Exporting error analysis to {export_error_analysis}')
                confidence = nn.functional.softmax(logits, dim=1)
                losses = nn.functional.cross_entropy(logits, labels, reduction='none')
                losses = nn.functional.normalize(losses, dim=0)
                original_sentences = batch[3]
                output_iter = []
                for conf, loss, label, seq in zip(
                        confidence, losses, labels, original_sentences):
                    seq = [self.id_to_word[word_id] for word_id in seq]
                    # prediction correctness
                    # label = (label.item() == torch.argmax(conf).item())
                    output_iter.append((conf.tolist(), label, loss, ' '.join(seq)))
                output_iter.sort(key=lambda tup: tup[0][0])

                with open(export_error_analysis, 'w') as file:
                    file.write(f'accuracy = {accuracy:.4f}\n\n')
                    file.write('(Dem confidence, GOP confidence)    '
                               'True Label    Normalized Loss\n')
                    for conf, label, loss, seq in output_iter:
                        file.write(f'({conf[0]:.2%}, {conf[1]:.2%})\t'
                                   f'{label}\t'
                                   f'{loss:.2%}\t'
                                   f'{seq}\n')
        self.train()
        return accuracy.item()


class BoWClassifierDataset(Dataset):

    def __init__(self, config: 'BoWClassifierConfig'):
        self.max_seq_length = config.max_sequence_length
        train_data_path = os.path.join(config.corpus_dir, 'train_data.pickle')
        print(f'Loading training data at {train_data_path}', flush=True)
        with open(train_data_path, 'rb') as file:
            preprocessed = pickle.load(file)
        # self.word_frequency: Dict[str, int] = preprocessed[2]
        sentences: List[Sentence] = preprocessed[3]
        test_sentences: List[Sentence] = preprocessed[4]
        if config.pretrained_embedding is not None:
            self.pretrained_embedding, self.word_to_id, self.id_to_word = (
                self.load_embeddings_from_plain_text(config.pretrained_embedding))
        else:
            self.pretrained_embedding = None
            self.word_to_id: Dict[str, int] = preprocessed[0]
            self.id_to_word: Dict[int, str] = preprocessed[1]

        # Prepare validation and test data
        # test_holdout = config.num_test_holdout
        # valid_holdout = config.num_valid_holdout
        # random.shuffle(sentences)
        # num_train = len(sentences) - test_holdout - valid_holdout
        self.train_data = sentences

        # test_data = sentences[num_train + test_holdout:]
        self.valid_batch = self.collate_bag_of_words(
            test_sentences, include_original_seq=True)
        # self.test_batch = self.collate_bag_of_words(
        #     test_data, include_original_seq=True)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index: int) -> Tuple[List[int], int]:
        sentence = self.train_data[index]
        return sentence.word_ids, sentence.party

    def collate_bag_of_words(self, batch, include_original_seq=False):
        party_labels = []
        flatten_sentences = []
        sent_starting_index = 0
        sent_starting_indicies = []
        for word_ids, party_label in batch:
            if len(word_ids) > self.max_seq_length:
                word_ids = word_ids[:self.max_seq_length]
            sent_starting_indicies.append(sent_starting_index)
            flatten_sentences.extend(word_ids)
            party_labels.append(party_label)
            sent_starting_index += len(word_ids)

        if include_original_seq:
            return (torch.LongTensor(flatten_sentences),
                    torch.LongTensor(sent_starting_indicies),
                    torch.LongTensor(party_labels),
                    [word_ids for word_ids, _ in batch])
        else:
            return (torch.LongTensor(flatten_sentences),
                    torch.LongTensor(sent_starting_indicies),
                    torch.LongTensor(party_labels))

    @staticmethod
    def load_embeddings_from_plain_text(path: str) -> Tuple[
            Matrix,
            Dict[str, int],
            Dict[int, str]]:
        """TODO turn off auto_caching"""
        id_generator = 0
        word_to_id: Dict[str, int] = {}
        id_to_word: Dict[int, str] = {}
        embeddings: List[float] = []  # cast to float when instantiating tensor
        print(f'Loading pretrained embeddings from {path}', flush=True)
        with open(path) as file:
            vocab_size, embed_size = map(int, file.readline().split())
            print(f'vocab_size = {vocab_size:,}, embed_size = {embed_size}')
            for line in file:
                line: List[str] = line.split()
                word = line[0]
                # vector = torch.Tensor(line[-embed_size:], dtype=torch.float32)
                embeddings.append(list(map(float, line[-embed_size:])))
                word_to_id[word] = id_generator
                id_to_word[id_generator] = word
                id_generator += 1
        embeddings = torch.FloatTensor(embeddings)
        return embeddings, word_to_id, id_to_word


class BoWClassifierExperiment(Experiment):

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
                flatten_sentences = batch[0].to(config.device)
                sent_starting_indicies = batch[1].to(config.device)
                party_labels = batch[2].to(config.device)
                self.optimizer.zero_grad()
                logits = self.model(flatten_sentences, sent_starting_indicies)
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
                        self.data.valid_batch,
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
class BoWClassifierConfig(ExperimentConfig):

    # Essential
    # corpus_dir='../data/processed/party_classification/44_Obama_doc_1e-5',
    corpus_dir: str = '../data/processed/party_classifier/UCSB_split_1e-5'
    output_dir: str = '../results/party_classifier/presidency/BoW/uniform_split_1e-5_.9dropout'
    device: torch.device = torch.device('cuda:0')

    # Hyperparmeters
    model: nn.Module = BoWClassifier
    embed_size: int = 300
    batch_size: int = 1024
    num_epochs: int = 50
    max_sequence_length: int = 30
    # pretrained_embedding: Optional[str] = None
    pretrained_embedding: Optional[str] = '../results/baseline/word2vec_president.txt'
    freeze_embedding: bool = False
    dropout_p: float = 0.9

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
    config = BoWClassifierConfig()

    config = BoWClassifierConfig()
    data = BoWClassifierDataset(config)
    dataloader = DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data.collate_bag_of_words,
        num_workers=config.num_dataloader_threads)
    model = config.model(config, data)
    optimizer = config.optimizer(model.parameters())
    lr_scheduler = config.lr_scheduler(optimizer)

    black_box = BoWClassifierExperiment(
        config, data, dataloader, model, optimizer, lr_scheduler)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    main()
