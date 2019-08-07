import functools
import pickle
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, List, Dict, Union, Optional
from typing import Counter as CounterType

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from party_classifier import NaiveWordClassifier
from skip_gram import SkipGramNegativeSampling
from utils.experiment import Experiment, ExperimentConfig
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor
from preprocessing.S4_export_training_corpus import Document

import random
random.seed(42)
torch.manual_seed(42)

class AdversarialDecomposer(nn.Module):

    def __init__(self, config: 'AdversarialConfig', data: 'AdversarialDataset'):
        super().__init__()
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        vocab_size = len(data.word_to_id)
        embed_size = config.embed_size
        self.device = config.device
        self.𝜀 = config.𝜀

        # Initialization
        if data.pretrained_embedding is not None:
            config.embed_size = data.pretrained_embedding.shape[1]
            self.center_embedding = nn.Embedding.from_pretrained(
                data.pretrained_embedding)
        else:
            self.center_embedding = nn.Embedding(
                vocab_size, embed_size, sparse=config.sparse_embedding)
            init_range = 1.0 / embed_size
            nn.init.uniform_(self.center_embedding.weight.data, -init_range, init_range)
        self.center_embedding.weight.requires_grad = not config.freeze_embedding

        # Adversarial Encoder
        self.encoder = nn.Linear(embed_size, config.hidden_size)
        # TODO dropout, non-linear layers here?
        self.dropout = nn.Dropout(p=config.dropout_p)
        self.deno_loss_weight = config.denotation_weight
        self.cono_loss_weight = config.connotation_weight

        # SGNS Denotation
        self.denotation_decoder = nn.Linear(config.hidden_size, embed_size)
        self.context_embedding = nn.Embedding(
            vocab_size, embed_size, sparse=config.sparse_embedding)
        nn.init.constant_(self.context_embedding.weight.data, 0)
        self.num_negative_samples = config.num_negative_samples
        SkipGramNegativeSampling.init_negative_sampling(
            self, data.word_frequency, data.word_to_id)

        # Party Classifier Connotation
        self.party_classifier = nn.Linear(
            config.hidden_size, config.num_prediction_classes)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.to(self.device)

    def forward(
            self,
            center_word_ids: Vector,
            context_word_ids: Matrix,
            party_label: Vector
            ) -> Tuple[Scalar, Scalar, Scalar]:
        encoded_dennotation = self.encoder_forward(center_word_ids)

        deno_loss = self.deno_forward(encoded_dennotation, context_word_ids)
        deno_loss = torch.clamp(deno_loss, self.𝜀, 10)

        cono_logits = self.cono_forward(encoded_dennotation)
        cono_loss = self.cross_entropy(cono_logits, party_label)
        cono_loss = torch.clamp(cono_loss, self.𝜀, 10)

        encoder_loss = (self.deno_loss_weight * deno_loss +
                        self.cono_loss_weight * cono_loss)
        encoder_loss = torch.clamp(encoder_loss, self.𝜀)
        return encoder_loss, deno_loss, cono_loss

    def encoder_forward(self, center_word_ids: Vector) -> Vector:
        """rename to decomposer forward?"""
        embeddings = self.center_embedding(center_word_ids)
        encoded_dennotation = self.encoder(embeddings)
        return self.dropout(encoded_dennotation)

    def deno_forward(
            self,
            encoded_dennotation: Matrix,
            context_ids: Vector
            ) -> Scalar:
        """
        Passing in decomposed/encoded denotation, not center_word_ids.
        """
        center_vectors = self.denotation_decoder(encoded_dennotation)
        # center_vectors = self.center_embedding(center_ids)
        context_vectors = self.context_embedding(context_ids)

        score = torch.sum(torch.mul(center_vectors, context_vectors), dim=1)
        score = torch.clamp(score, -10, 10)
        score = -nn.functional.logsigmoid(score)

        batch_size = context_ids.shape[0]
        neg_context_ids = self.negative_sampling_dist.sample(
            (batch_size, self.num_negative_samples))
        neg_context_vectors = self.context_embedding(neg_context_ids)

        neg_score = torch.bmm(neg_context_vectors, center_vectors.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, -10, 10)
        neg_score = -torch.sum(nn.functional.logsigmoid(-neg_score), dim=1)
        return torch.mean(score + neg_score)

    def cono_forward(self, encoded_dennotation: Matrix) -> Vector:
        logits = self.party_classifier(encoded_dennotation)
        return logits

    def predict_connotation(self, word_ids: Vector) -> Vector:
        self.eval()
        with torch.no_grad():
            encoded = self.encoder_forward(word_ids)
            logits = self.cono_forward(encoded)
            confidence = nn.functional.softmax(logits, dim=1)
        self.train()
        return confidence

    def connotation_accuracy(
            self,
            batch: Tuple,
            export_error_analysis: Optional[str] = None
            ) -> float:
        word_ids = batch[0].to(self.device)
        labels = batch[2].to(self.device)

        confidence = self.predict_connotation(word_ids)
        predictions = confidence.argmax(dim=1)
        correct_indicies = predictions.eq(labels)
        accuracy = correct_indicies.float().mean().item()

        # conf = [c[1].item() for c in confidence]
        # high_conf = [
        #     f'{self.id_to_word[word_ids[i].item()]}={c:.4f}'
        #     for i, c in enumerate(conf) if c > .9]
        # if len(high_conf) > 0:
        #     tqdm.write(', '.join(high_conf))
        return accuracy

    def all_vocab_connotation(
            self,
            export_path: Optional[str] = None
            ) -> Vector:
        """Inspect the decomposed vectors"""
        all_vocab_ids = torch.arange(
            len(self.word_to_id), dtype=torch.long, device=self.device)
        confidence = self.predict_connotation(all_vocab_ids)
        if not export_path:
            return confidence

        tqdm.write(f'Exporting error analysis to {export_path}')
        output = []
        for conf, word_id in zip(confidence, all_vocab_ids):  # type: ignore
            word = self.id_to_word[word_id.item()]
            output.append((conf.tolist(), word))
        output.sort(key=lambda tup: tup[0][0])  # ascending GOP confidence
        # self.accuracy_at_confidence_plot(output)  # TODO

        with open(export_path, 'w') as file:
            # file.write(f'accuracy = {accuracy:.2%}\n\n')
            file.write('(Dem confidence, GOP confidence)\n')
            for conf, word in output:
                file.write(f'({conf[0]:.2%}, {conf[1]:.2%})\t{word}\n')

    def export_decomposed_embedding(
            self,
            export_path: Optional[str] = None,
            tensorboard: bool = False
            ) -> None:
        """for querying nearest neighbors & visualization"""
        all_vocab_ids = torch.arange(
            len(self.word_to_id), dtype=torch.long, device=self.device)
        self.eval()
        with torch.no_grad():
            decomposed = self.encoder_forward(all_vocab_ids)
        self.train()

        if export_path:
            raise NotImplementedError
        if tensorboard:
            embedding_labels = [
                self.id_to_word[word_id]
                for word_id in all_vocab_ids]
            self.tensorboard.add_embedding(
                decomposed, embedding_labels, global_step=0)
        return decomposed


# TODO move load pretrained embedding to utils
# TODO try torch.chunk, split
class AdversarialDataset(Dataset):

    def __init__(self, config: 'AdversarialConfig'):
        corpus_path = os.path.join(config.input_dir, 'train_data.pickle')
        with open(corpus_path, 'rb') as corpus_file:
            preprocessed = pickle.load(corpus_file)

        self.pretrained_embedding: Optional[Matrix]
        if config.pretrained_embedding is not None:
            self.pretrained_embedding, self.word_to_id, self.id_to_word = (
                self.load_embeddings_from_plain_text(config.pretrained_embedding))
        else:
            self.pretrained_embedding = None
            self.word_to_id = preprocessed[0]
            self.id_to_word = preprocessed[1]

        self.word_frequency: CounterType[str] = preprocessed[2]
        speeches: List[Document] = preprocessed[3]
        if config.debug_subset_corpus:
            speeches = speeches[:config.debug_subset_corpus]
        self.speeches: List[Document] = speeches
        self.window_radius = config.window_radius

        self.total_num_epochs = config.num_epochs
        self.min_doc_length = config.min_doc_length
        self.conf_threshold_schedule = iter(
            config.neutral_word_subsampling_threshold_schedule)
            # .to(config.device))

    def __len__(self) -> int:
        return len(self.speeches)

    def __getitem__(self, index: int) -> Tuple[Vector, Vector, Vector]:
        """
        parse one document into a List[skip-grams], where each skip-gram
        is a Tuple[center_id, List[context_ids]]
        """
        doc: List[int] = self.speeches[index].word_ids
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
        return (
            torch.tensor(center_word_ids),
            torch.tensor(context_word_ids),
            torch.full(
                (len(context_word_ids),),
                self.speeches[index].party,
                dtype=torch.int64))

    @staticmethod
    def collate(
            faux_batch: List[Tuple[Vector, Vector, Vector]]
            ) -> Tuple[Vector, Vector, Vector]:
        center_word_ids = torch.cat([center for center, _, _ in faux_batch])
        context_word_ids = torch.cat([context for _, context, _ in faux_batch])
        labels = torch.cat([label for _, _, label in faux_batch])
        shuffle = torch.randperm(len(labels))
        return center_word_ids[shuffle], context_word_ids[shuffle], labels[shuffle]
        # return center_word_ids, context_word_ids, labels

    def subsample_neutral_words(
            self,
            vocab_confidence: Vector,
            current_epoch: int,
            power: float
            ) -> int:
        """ P(keep word) = (prediction_conf / conf_threshold) ^ power """
        conf_threshold = next(self.conf_threshold_schedule)
        prediction_conf: Vector = vocab_confidence.max(dim=1).values
        keep_prob = torch.pow(prediction_conf / conf_threshold, power)
        keep_prob = keep_prob.tolist()  # significantly improves performance

        subsampled_speeches: List[Document] = []
        original_num_words = 0
        subsampled_num_words = 0
        num_discarded_doc = 0
        tqdm.write(
            f'Connotation confidence lower bound = {conf_threshold:.2%}, '
            f'power = {power}')
        progress_bar = tqdm(
            self.speeches, desc='Subsampling neutral words', mininterval=1)
        for doc in progress_bar:
            subsampled_word_ids = [
                word_id
                for word_id in doc.word_ids
                if random.random() < keep_prob[word_id]]
            original_num_words += len(doc.word_ids)
            if len(subsampled_word_ids) > self.min_doc_length:
                subsampled_speeches.append(
                    Document(subsampled_word_ids, doc.party))
                subsampled_num_words += len(subsampled_word_ids)
            else:
                num_discarded_doc += 1
        self.speeches = subsampled_speeches
        tqdm.write(
            f'Total number of words subsampled from {original_num_words:,} '
            f'to {subsampled_num_words:,}; where {num_discarded_doc} documents '
            f'are discarded because they are too short.')
        return subsampled_num_words

    @staticmethod
    def load_embeddings_from_plain_text(in_path: str) -> Tuple[
            Matrix,
            Dict[str, int],
            Dict[int, str]]:
        id_generator = 0
        word_to_id: Dict[str, int] = {}
        id_to_word: Dict[int, str] = {}
        embeddings: List[List[float]] = []  # cast to float when instantiating tensor
        print(f'Loading pretrained embeddings from {in_path}', flush=True)
        with open(in_path) as file:
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


class AdversarialExperiment(Experiment):

    def __init__(
            self,
            config: ExperimentConfig,
            data: Dataset,
            dataloader: DataLoader,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler._LRScheduler):
        super().__init__(config, data, dataloader, model, optimizer, lr_scheduler)
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)  # param.data)

        if config.freeze_embedding:
            self.encoder_optimizer = torch.optim.Adam(
                model.encoder.parameters())
        else:
            self.encoder_optimizer = torch.optim.Adam(
                list(model.center_embedding.parameters()) +
                list(model.encoder.parameters()))

        self.cono_optimizer = torch.optim.Adam(
            model.party_classifier.parameters())
        self.deno_optimizer = torch.optim.Adam(
            list(model.denotation_decoder.parameters()) +
            list(model.context_embedding.parameters()))

        self.custom_stats_format = (
            '𝓁encoder = {encoder_loss:.3f}\t'
            f'({config.denotation_weight}) '
            '𝓁deno = {dennotation_loss:.3f}\t'
            f'({config.connotation_weight}) '
            '𝓁cono = {connotation_loss:.3f}\t'
            'train accuracy = {training_accuracy:.2%}')

        # self.tensorboard.add_custom_scalars({
        #     'Dennotation': {
        #         'loss': ['Multiline', ['dennotation_loss']],
        #     }
        #     'Connotation': {
        #         'loss': ['Multiline', ['connotation_loss']],
        #         'training_accuracy': ['Multiline', ['training_accuracy']]
        #     },
        #     'Encoder': {
        #         'loss': ['Multiline', ['encoder_loss']]
        #     }
        # })

    # @staticmethod
    # def reload_everything(path: str, device: torch.device):
    #     print(f'Reloading model and config from {path}')
    #     payload = torch.load(path, map_location=device)
    #     return AdversarialExperiment(
    #                 payload['config'], data, dataloader, payload['model'],
    #                 optimizer=None, lr_scheduler=None)


    def safe_train(self, batch: Tuple) -> Tuple[float, float, float]:
        center_word_ids = batch[0].to(self.device)
        context_word_ids = batch[1].to(self.device)
        party_labels = batch[2].to(self.device)

        self.model.zero_grad()
        l_encoder, l_deno, l_cono = self.model(
            center_word_ids, context_word_ids, party_labels)
        l_encoder.backward()
        self.encoder_optimizer.step()

        self.model.zero_grad()
        l_encoder, l_deno, l_cono = self.model(
            center_word_ids, context_word_ids, party_labels)
        l_deno.backward()
        self.deno_optimizer.step()

        self.model.zero_grad()
        l_encoder, l_deno, l_cono = self.model(
            center_word_ids, context_word_ids, party_labels)
        l_cono.backward()
        self.cono_optimizer.step()
        return l_encoder.item(), l_deno.item(), l_cono.item()

    def _train(self, batch: Tuple) -> Tuple[float, float, float]:
        center_word_ids = batch[0].to(self.device)
        context_word_ids = batch[1].to(self.device)
        party_labels = batch[2].to(self.device)

        self.model.zero_grad()
        l_encoder, l_deno, l_cono = self.model(
            center_word_ids, context_word_ids, party_labels)
        l_encoder.backward(retain_graph=True)
        self.encoder_optimizer.step()  # ???

        self.model.zero_grad()
        l_deno.backward(retain_graph=True)
        self.deno_optimizer.step()

        self.model.zero_grad()
        l_cono.backward()
        self.cono_optimizer.step()

        return l_encoder.item(), l_deno.item(), l_cono.item()

    def fast_train(self, batch: Tuple) -> Tuple[float, float, float]:
        center_word_ids = batch[0].to(self.device)
        context_word_ids = batch[1].to(self.device)
        party_labels = batch[2].to(self.device)

        l_encoder, l_deno, l_cono = self.model(
            center_word_ids, context_word_ids, party_labels)

        self.model.zero_grad()
        l_encoder.backward()  # computes all gradients

        self.encoder_optimizer.step()  # apply gradients accordingly
        # self.encoder_optimizer.zero_grad()
        self.deno_optimizer.step()
        # self.deno_optimizer.zero_grad()
        self.cono_optimizer.step()
        return l_encoder.item(), l_deno.item(), l_cono.item()

    def train(self) -> None:
        config = self.config
        self.device = config.device
        epoch_pbar = tqdm(range(1, config.num_epochs + 1), desc='Epochs')
        for epoch_index in epoch_pbar:
            batch_pbar = tqdm(
                enumerate(self.dataloader), total=len(self.dataloader),
                mininterval=1, desc='Batches')
            for batch_index, batch in batch_pbar:
                l_encoder, l_deno, l_cono = self.safe_train(batch)

                if batch_index % config.update_tensorboard == 0:
                    stats = {
                        'encoder_loss': l_encoder,
                        'dennotation_loss': l_deno,
                        'connotation_loss': l_cono,
                        'training_accuracy': self.model.connotation_accuracy(batch)
                    }
                    self.update_tensorboard(stats)
                if batch_index % config.print_stats == 0:
                    self.print_stats(epoch_index, batch_index, stats)
            # End Batches
            self.print_timestamp()
            # self.lr_scheduler.step()  # NOTE
            if config.export_error_analysis:
                if (epoch_index % config.export_error_analysis == 0
                        or epoch_index == 1):
                    export_path = os.path.join(
                        config.output_dir,
                        f'error_analysis_epoch{epoch_index}.txt')
                    self.model.all_vocab_connotation(export_path)

            # Discard partisan neutral words
            vocab_party_confidence = self.model.all_vocab_connotation()
            subsampled_num_words = self.data.subsample_neutral_words(
                vocab_party_confidence, epoch_index, config.subsampling_power)
            self.update_tensorboard(
                {'num_words_subsampled_corpus': subsampled_num_words})

            if config.auto_save_every_epoch or epoch_index == config.num_epochs:
                self.save_everything(os.path.join(
                    config.output_dir, f'epoch{epoch_index}.pt'))
        # End Epochs
        print('\n✅ Training Complete')


@dataclass
class AdversarialConfig():
    # Essential
    input_dir: str = '../data/processed/adversarial/44_Obama_1e-5'
    output_dir: str = '../results/adversarial/2000s/p8_.55to.75/d1_c-1'
    device: torch.device = torch.device('cuda:0')
    debug_subset_corpus: Optional[int] = None
    num_dataloader_threads: int = 0

    # Hyperparameters
    denotation_weight: float = 1  # denotation weight 𝛿
    connotation_weight: float = -1  # connotation weight 𝛾
    batch_size: int = 8
    embed_size: int = 300
    hidden_size: int = 300
    window_radius: int = 5  # context_size = 2 * window_radius
    num_negative_samples: int = 10
    dropout_p: float = 0
    num_epochs: int = 10

    neutral_word_subsampling_threshold_schedule: Vector = torch.linspace(
        0.55, 0.75, steps=num_epochs)
    subsampling_power: float = 8  # float('inf') for deterministic sampling
    min_doc_length: int = 2  # for subsampling partisan neutral words

    # Model Construction
    model: nn.Module = AdversarialDecomposer
    pretrained_embedding: Optional[str] = None
    freeze_embedding: bool = True
    num_prediction_classes: int = 2
    sparse_embedding: bool = False  # faster if used with sparse optimizer
    𝜀: float = 1e-5

    # Optimizer
    optimizer: functools.partial = functools.partial(
        torch.optim.Adam,
        lr=1e-3,
        # weight_decay=1e-3
    )
    lr_scheduler: functools.partial = functools.partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=10,
        gamma=0.1
    )

    update_tensorboard: int = 100  # per batch
    print_stats: int = 1000  # per batch
    export_error_analysis: Optional[int] = None  # per epoch
    reload_path: Optional[str] = None
    clear_tensorboard_log_in_output_dir: bool = True
    delete_all_exisiting_files_in_output_dir: bool = False
    auto_save_every_epoch: bool = False
    auto_save_before_quit: bool = True
    tensorboard_embedding_projector: bool = False


def main() -> None:
    # config = AdversarialConfig()
    # config = AdversarialConfig(
    #     output_dir='../results/adversarial/Obama/p8_.55to.75/thawed_embed/d-0.1_c1',
    #     denotation_weight=-0.1,
    #     connotation_weight=1,
    #     device=torch.device('cuda:1')
    # )
    # config = AdversarialConfig(
    #     output_dir='../results/adversarial/Obama/p8_.5to.7/thawed_d0_c1',
    #     denotation_weight=0,
    #     connotation_weight=1,
    #     neutral_word_subsampling_threshold_schedule=(
    #         [.5] * 3 + [.55, .575, .6, .625, .65, .675, .7]
    #     ),
    #     freeze_embedding=False,
    #     device=torch.device('cuda:0')
    # )
    config = AdversarialConfig(
        output_dir='../results/adversarial/Obama/p8_fixed.65/thawed_w2v_d0_c1',
        denotation_weight=0,
        connotation_weight=1,
        neutral_word_subsampling_threshold_schedule=(
            [.5] * 4 + [.65] * 6
        ),
        pretrained_embedding='../results/baseline/word2vec_Obama.txt',
        freeze_embedding=False,
        device=torch.device('cuda:0')
    )

    data = AdversarialDataset(config)
    dataloader = DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data.collate,
        num_workers=config.num_dataloader_threads)
    model = config.model(config, data)
    # optimizer = config.optimizer(model.parameters())
    # lr_scheduler = config.lr_scheduler(optimizer)
    black_box = AdversarialExperiment(
        config, data, dataloader, model, optimizer=None, lr_scheduler=None)

    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    main()
