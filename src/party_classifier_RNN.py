import functools
import pickle
import random
import os
from dataclasses import dataclass
from typing import Tuple, NamedTuple, List, Dict, Optional

import torch
from torch import nn
from torch.nn.utils import rnn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.experiment import Experiment, ExperimentConfig
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor
from preprocessing.S4_export_training_corpus import Sentence
# class Sentence(NamedTuple):
#     word_ids: List[int]
#     party: int  # Dem -> 0; GOP -> 1.

random.seed(0)
torch.manual_seed(0)

class LSTMClassifier(nn.Module):

    def __init__(
            self,
            config: 'RNNClassifierConfig',
            data: 'RNNClassifierDataset'):
        super().__init__()
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_RNN_layers
        self.device = config.device
        if getattr(data, 'pretrained_embedding') is not None:
            config.embed_size = data.pretrained_embedding.shape[1]
            self.embedding = nn.Embedding.from_pretrained(data.pretrained_embedding)
        else:
            self.embedding = nn.Embedding(len(self.word_to_id), config.embed_size)
            init_range = 1.0 / config.embed_size
            nn.init.uniform_(self.embedding.weight.data, -init_range, init_range)
            # nn.init.normal_(self.embedding.weight.data, 0, init_range)
        self.embedding.weight.requires_grad = not config.freeze_embedding

        self.LSTM = nn.LSTM(
            config.embed_size, self.hidden_size, num_layers=self.num_layers)

        self.dropout = nn.Dropout(p=config.dropout_p)
        self.linear_classifier = nn.Linear(self.hidden_size, config.num_prediction_classes)

        self.to(self.device)

    def forward(self, padded_sequences: Matrix, seq_lengths: Vector) -> Matrix:
        """
        LSTM with fixed attention
        padded_sequences: (max_seq_len, batch_size)
        """
        seq_embed = self.embedding(padded_sequences)
        packed_seq = rnn.pack_padded_sequence(
            seq_embed, seq_lengths, enforce_sorted=False)

        batch_size = len(seq_lengths)
        init_state = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))
        packed_output, (final_hidden, final_cell) = self.LSTM(packed_seq, init_state)
        # output = (batch size, seq len, hid dim * num directions)
        # hidden = (num layers * num directions, batch size, hid dim)
        logits = self.linear_classifier(self.dropout(final_hidden[-1]))
        return logits

    def accuracy(
            self,
            batch: Tuple,
            export_error_analysis: Optional[str] = None
            ) -> float:
        padded_sequences = batch[0].to(self.device)
        seq_lengths = batch[1].to(self.device)
        labels = batch[2].to(self.device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(padded_sequences, seq_lengths)
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
                output_iter = []
                for conf, loss, label, (index, seq_len) in zip(
                        confidence, losses, labels, enumerate(seq_lengths)):
                    seq = [self.id_to_word[word_id.item()]
                           for word_id in padded_sequences[0:seq_len, index]]
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


class LSTMAttentionClassifier(LSTMClassifier):

    def __init__(
            self,
            config: 'RNNClassifierConfig',
            data: 'RNNClassifierDataset'):
        nn.Module.__init__(self)
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_RNN_layers
        self.device = config.device

        if data.pretrained_embedding is not None:
            config.embed_size = data.pretrained_embedding.shape[1]
            self.embedding = nn.Embedding.from_pretrained(data.pretrained_embedding)
        else:
            self.embedding = nn.Embedding(len(self.word_to_id), config.embed_size)
            init_range = 1.0 / config.embed_size
            nn.init.uniform_(self.embedding.weight.data, -init_range, init_range)
            # nn.init.normal_(self.embedding.weight.data, 0, init_range)
        self.embedding.weight.requires_grad = config.freeze_embedding

        self.LSTM = nn.LSTM(
            config.embed_size, self.hidden_size, num_layers=self.num_layers)

        self.dropout = nn.Dropout(p=config.dropout_p)
        # self.attention = nn.Linear(hidden_size, 50).to(device)  # learnable attn
        self.linear_classifier = nn.Linear(self.hidden_size, config.num_prediction_classes)

        self.to(self.device)

    def forward(self, padded_sequences: Matrix, seq_lengths: Vector) -> Matrix:
        """
        LSTM with fixed attention
        padded_sequences: (max_seq_len, batch_size)
        """
        seq_embed = self.embedding(padded_sequences)
        packed_seq = rnn.pack_padded_sequence(
            seq_embed, seq_lengths, enforce_sorted=False)

        batch_size = len(seq_lengths)
        init_state = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))
        packed_output, (final_hidden, final_cell) = self.LSTM(packed_seq, init_state)
        padded_output, seq_lengths = rnn.pad_packed_sequence(
            packed_output, batch_first=True)
        # output = (batch size, seq len, hid dim * num directions)
        # hidden = (num layers * num directions, batch size, hid dim)

        # bsh, bh1 -> bs  # Fixed attention
        # final_hidden = final_hidden.squeeze(0)  # only need last layer
        # attention_weights = torch.einsum('bsh,bh->bs', output, hidden_state)
        final_hidden = final_hidden[-1].unsqueeze(2)
        attn_weights = torch.bmm(padded_output, final_hidden).squeeze(2)
        attn_dist = nn.functional.softmax(attn_weights, dim=1)

        # bsh/bhs, bs1 -> bh
        padded_output = padded_output.transpose(1, 2)
        attn_dist = attn_dist.unsqueeze(2)
        attended_output = torch.bmm(padded_output, attn_dist).squeeze(2)

        # # logits = self.MLP_classifier(attended_output)
        logits = self.linear_classifier(self.dropout(attended_output))
        return logits


class GRUAttentionClassifier(LSTMAttentionClassifier):

    def __init__(
            self,
            word_to_id: Dict[str, int],
            id_to_word: Dict[int, str],
            embed_size: int,
            hidden_size: int,
            num_layers: int,
            dropout_p: float,
            num_prediction_classes: int,
            device: torch.device
            ) -> None:
        nn.Module.__init__(self)
        self.vocab_size = len(word_to_id)
        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        # init_range = 1.0 / embed_size
        # nn.init.uniform_(
        #     self.embedding.weight.data, -init_range, init_range)

        self.GRU = nn.GRU(embed_size, hidden_size, num_layers=num_layers)

        # self.MLP_classifier = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(hidden_size, num_prediction_classes)
        # ).to(device)
        # self.attention = nn.Linear(hidden_size, 50).to(device)  # TODO max seq len
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear_classifier = nn.Linear(hidden_size, num_prediction_classes)

        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.to(device)

    def forward(self, padded_sequences: Matrix, seq_lengths: Vector) -> Matrix:
        """
        padded_sequences: (max_seq_len, batch_size)
        """
        seq_embed = self.embedding(padded_sequences)
        packed_seq = rnn.pack_padded_sequence(
            seq_embed, seq_lengths, enforce_sorted=False)

        batch_size = len(seq_lengths)
        init_hidden_state = torch.zeros(
            self.num_layers, batch_size, self.hidden_size).to(self.device)

        packed_output, final_hidden = self.GRU(packed_seq, init_hidden_state)
        # padded_output, seq_lengths = rnn.pad_packed_sequence(
        #     packed_output, batch_first=True)
        # output = (batch size, seq len, hid dim * num directions)
        # hidden = (num layers * num directions, batch size, hid dim)

        # Attention
        # bsh, bh1 -> bs
        # final_hidden = final_hidden.squeeze(0)  # only need last layer
        # attention_weights = torch.einsum('bsh,bh->bs', output, hidden_state)
        # final_hidden = final_hidden[-1].unsqueeze(2)
        # attn_weights = torch.bmm(padded_output, final_hidden).squeeze(2)
        # attn_dist = nn.functional.softmax(attn_weights, dim=1)

        # # Lernable attnetion quierd by final hidden state, fixed sequence length
        # # final_hidden = final_hidden.squeeze(0)  # bh
        # # attn_weights = self.attention(final_hidden)  # bs
        # # attn_dist = nn.functional.softmax(attn_weights, dim=1)

        # # bsh/bhs, bs1 -> bh
        # padded_output = padded_output.transpose(1, 2)
        # attn_dist = attn_dist.unsqueeze(2)
        # attended_output = torch.bmm(padded_output, attn_dist).squeeze(2)

        # logits = self.MLP_classifier(attended_output)
        # logits = self.linear_classifier(attended_output)

        features = self.dropout(final_hidden[-1])
        logits = self.linear_classifier(features)
        return logits


class RNNClassifierDataset(Dataset):

    def __init__(self, config: 'RNNClassifierConfig'):
        train_data_path = os.path.join(config.corpus_dir, 'train_data.pickle')
        print(f'Loading training data at {train_data_path}', flush=True)
        with open(train_data_path, 'rb') as file:
            preprocessed = pickle.load(file)
        # self.word_frequency: Dict[str, int] = preprocessed[2]
        sentences: List[Sentence] = preprocessed[3]
        if config.pretrained_embedding is not None:
            self.pretrained_embedding, self.word_to_id, self.id_to_word = (
                self.load_embeddings_from_plain_text(config.pretrained_embedding))
        else:
            self.pretrained_embedding = None
            self.word_to_id: Dict[str, int] = preprocessed[0]
            self.id_to_word: Dict[int, str] = preprocessed[1]
        self.max_seq_length = config.max_sequence_length

        # Prepare validation and test data
        test_holdout = config.num_test_holdout
        valid_holdout = config.num_valid_holdout
        random.shuffle(sentences)
        num_train_samples = len(sentences) - test_holdout - valid_holdout
        self.train_data = sentences[:num_train_samples]
        # self.train_data = sentences[:1024]  # for debugging

        valid_batch = []
        for speech in sentences[num_train_samples:num_train_samples + valid_holdout]:
            if len(speech.word_ids) > self.max_seq_length:
                valid_batch.append((
                    speech.word_ids[:self.max_seq_length],
                    self.max_seq_length,
                    speech.party))
            else:
                valid_batch.append((speech.word_ids, len(speech.word_ids), speech.party))
        self.validation_batch = self.pad_sequences(valid_batch)

        test_batch = []
        for speech in sentences[num_train_samples + valid_holdout:]:
            if len(speech.word_ids) > self.max_seq_length:
                test_batch.append((
                    speech.word_ids[:self.max_seq_length],
                    self.max_seq_length,
                    speech.party))
            else:
                test_batch.append((speech.word_ids, len(speech.word_ids), speech.party))
        self.test_batch = self.pad_sequences(test_batch)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index: int) -> Tuple[List[int], int, int]:
        speech: Sentence = self.train_data[index]
        if len(speech.word_ids) > self.max_seq_length:
            return (speech.word_ids[:self.max_seq_length],
                    self.max_seq_length,
                    speech.party)
        else:
            return speech.word_ids, len(speech.word_ids), speech.party

    @staticmethod
    def pad_sequences(batch):
        sequences = [torch.LongTensor(word_ids) for word_ids, _, _ in batch]
        seq_lengths = [seq_len for _, seq_len, _ in batch]
        labels = [label for _, _, label in batch]
        return (rnn.pad_sequence(sequences),
                torch.LongTensor(seq_lengths),
                torch.LongTensor(labels))

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


class RNNClassifierExperiment(Experiment):

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
                padded_sequences = batch[0].to(config.device)
                seq_lengths = batch[1].to(config.device)
                labels = batch[2].to(config.device)

                self.model.zero_grad()
                logits = self.model(padded_sequences, seq_lengths)
                loss = cross_entropy(logits, labels)
                loss.backward()
                self.optimizer.step()

                if batch_index % config.update_tensorboard_per_batch == 0:
                    batch_accuracy = self.model.accuracy(batch)
                    tqdm.write(f'Epoch {epoch_index}, Batch {batch_index:,}:\t'
                               f'Loss = {loss.item():.5f}\t'
                               f'Train Accuracy = {batch_accuracy:.2%}')

                    self.tensorboard.add_scalar(
                        'training loss', loss.item(), tb_global_step)
                    self.tensorboard.add_scalar(
                        'training accuracy', batch_accuracy, tb_global_step)
                    tb_global_step += 1

                # if (batch_index % config.print_stats_per_batch) == 0:
                #     self.print_stats(loss.item(), epoch_index, batch_index)
            # end batch

            self.lr_scheduler.step()
            valid_accuracy = self.model.accuracy(self.data.validation_batch)
            tqdm.write(f'Epoch {epoch_index}  '
                       f'Validation Accuracy = {valid_accuracy:.2%}\n\n')
            self.tensorboard.add_scalar(
                'validation accuracy', valid_accuracy, epoch_index)
            if config.export_error_analysis:
                if epoch_index == 1 or epoch_index % 10 == 0:
                    valid_accuracy = self.model.accuracy(
                        self.data.validation_batch,
                        export_error_analysis=os.path.join(
                            config.output_dir,
                            f'error_analysis_epoch{epoch_index}.txt'))

            if config.auto_save_every_epoch or epoch_index == config.num_epochs:
                self.save_state_dict(epoch_index, tb_global_step)
        # end epoch
        test_accuracy = self.model.accuracy(self.data.test_batch)
        self.tensorboard.add_scalar('test accuracy', test_accuracy, global_step=0)
        print(f'Test Accuracy = {test_accuracy:.2%}')
        print('\n✅ Training Complete')


@dataclass
class RNNClassifierConfig(ExperimentConfig):

    # Essential
    # corpus_dir: str = '../data/processed/party_classifier/UCSB_1e-5'
    # output_dir: str = '../results/party_classifier/presidency/uniform_1e-5'
    corpus_dir: str = '../data/processed/party_classifier/44_Obama_len30sent_1e-3'
    output_dir: str = '../results/party_classifier/RNN/Obama_uniform_.9drop_1e-3'
    device: torch.device = torch.device('cuda:1')

    # Hyperparmeters
    model: nn.Module = LSTMClassifier
    hidden_size: int = 512
    pretrained_embedding: Optional[str] = None
    # pretrained_embedding: Optional[str] = '../results/baseline/word2vec_president.txt'
    embed_size: int = 300
    freeze_embedding: bool = False
    dropout_p: float = 0.9
    batch_size: int = 1000
    num_epochs: int = 50
    num_RNN_layers: int = 1
    max_sequence_length: int = 30

    # Optimizer
    optimizer: functools.partial = functools.partial(
        torch.optim.Adam,
        lr=1e-3,
        # weight_decay=0.2
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
    export_error_analysis: bool = True  # TODO change to epoch frequency

    # Housekeeping
    auto_save_before_quit: bool = True
    save_to_tensorboard_embedding_projector: bool = False
    update_tensorboard_per_batch: int = 1_000
    print_stats_per_batch: int = 1_000


def main() -> None:
    config = RNNClassifierConfig()
    data = RNNClassifierDataset(config)
    dataloader = DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data.pad_sequences,
        num_workers=config.num_dataloader_threads)
    model = config.model(config, data)
    optimizer = config.optimizer(model.parameters())
    lr_scheduler = config.lr_scheduler(optimizer)

    # model = nn.DataParallel(model, device_ids=[0, 1])  # late-stage capitalism

    black_box = RNNClassifierExperiment(
        config, data, dataloader, model, optimizer, lr_scheduler)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()

if __name__ == '__main__':
    main()

    # import cProfile
    # cProfile.run('main()', sort='cumulative')
