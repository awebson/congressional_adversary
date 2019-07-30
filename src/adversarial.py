import functools
import pickle
import os
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

# Scalar = torch.Tensor
# Vector = torch.Tensor
# Matrix = torch.Tensor
EncoderConfig = Union['DenotationEncoderConfig', 'ConnotationEncoderConfig']

torch.manual_seed(42)

class DenotationEncoder(nn.Module):

    def __init__(self, config: EncoderConfig, data: 'AdversarialDataset'):
        """
        𝛿, 𝛾 ∈ (0, 1)
        """
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
        self.deno_loss_weight = config.denotation_weight
        self.cono_loss_weight = config.connotation_weight

        # SGNS Denotation
        self.denotation_to_center_vec = nn.Linear(config.hidden_size, embed_size)
        self.context_embedding = nn.Embedding(
            vocab_size, embed_size, sparse=config.sparse_embedding)
        nn.init.constant_(self.context_embedding.weight.data, 0)
        self.num_negative_samples = config.num_negative_samples
        SkipGramNegativeSampling.init_negative_sampling(
            self, data.word_frequency, data.word_to_id)

        # Party Classifier Connotation
        self.dropout = nn.Dropout(p=config.dropout_p)
        self.linear_classifier = nn.Linear(
            config.hidden_size, config.num_prediction_classes)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.to(self.device)

    def forward(
            self,
            center_word_ids: Vector,
            context_word_ids: Matrix,
            party_label: Vector
            ) -> Tuple[Scalar, Scalar, Scalar]:
        embeddings = self.center_embedding(center_word_ids)
        dennotation = self.encoder(embeddings)

        center_vectors = self.denotation_to_center_vec(dennotation)
        deno_loss = self.deno_forward(center_vectors, context_word_ids)

        cono_logits = self.cono_forward(dennotation)
        cono_loss = self.cross_entropy(cono_logits, party_label)
        cono_loss = torch.clamp(cono_loss, self.𝜀, 5)

        encoder_loss = (self.deno_loss_weight * deno_loss +
                        self.cono_loss_weight * cono_loss)
        encoder_loss = torch.clamp(encoder_loss, self.𝜀)
        return encoder_loss, deno_loss, cono_loss

    def encoder_forward():
        pass

    def deno_forward(
            self,
            center_vectors: Matrix,
            context_ids: Vector
            ) -> Scalar:
        """
        Passing in (decomposed) center vectors, not center_word_ids.
        """
        # center_vectors = self.center_embedding(center_ids)
        context_vectors = self.context_embedding(context_ids)

        score = torch.sum(torch.mul(center_vectors, context_vectors), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -nn.functional.logsigmoid(score)

        batch_size = context_ids.shape[0]
        neg_context_ids = self.negative_sampling_dist.sample(
            (batch_size, self.num_negative_samples))
        neg_context_vectors = self.context_embedding(neg_context_ids)

        neg_score = torch.bmm(neg_context_vectors, center_vectors.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(nn.functional.logsigmoid(-neg_score), dim=1)
        return torch.mean(score + neg_score)


    def cono_forward(self, features: Vector) -> Vector:
        logits = self.linear_classifier(self.dropout(features))
        return logits

    def connotation_accuracy(
            self,
            batch: Tuple,
            export_error_analysis: Optional[str] = None
            ) -> float:
        word_ids = batch[0].to(self.device)
        labels = batch[2].to(self.device)
        self.eval()
        with torch.no_grad():
            encoded = self.encoder(self.center_embedding(word_ids))
            logits = self.cono_forward(encoded)
            predictions = logits.argmax(dim=1)
            accuracy = predictions.eq(labels).float().mean().item()

            conf = [c[1].item() for c in nn.functional.softmax(logits, dim=1)]
            high_conf = [
                f'{self.id_to_word[word_ids[i].item()]}={c:.4f}'
                for i, c in enumerate(conf) if c > .9]
            # if len(high_conf) > 0:
            #     tqdm.write(', '.join(high_conf))

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

    def all_vocab_confidence(self, export_error_analysis: Optional[str] = None
            ) -> Vector:
        all_vocab_word_ids = torch.arange(dtype, device)
        logits = cono_forward(encoder_forward(all_vocab_word_ids))
        confidence = nn.functional.softmax(logits, dim=1)
        return confidence


class ConnotationEncoder(nn.Module):

    𝜀 = 1e-5

    def __init__(self, config: 'AdversarialConfig', data: 'AdversarialDataset'):
        """
        𝛿, 𝛾 ∈ (0, 1)
        """
        super().__init__()
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word

        # SGNS
        vocab_size = len(data.word_to_id)
        embed_size = config.embed_size
        self.num_negative_samples = config.num_negative_samples
        self.device = config.device

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

        self.context_embedding = nn.Embedding(
            vocab_size, embed_size, sparse=config.sparse_embedding)
        nn.init.constant_(self.context_embedding.weight.data, 0)

        SkipGramNegativeSampling.init_negative_sampling(
            self, data.word_frequency, data.word_to_id)

        # classifier
        self.dropout = nn.Dropout(p=config.dropout_p)
        self.linear_classifier = nn.Linear(config.embed_size, config.num_prediction_classes)
        self.cross_entropy = nn.CrossEntropyLoss()

        # # two discriminators
        # self.connotation = NaiveWordClassifier(config, data)
        # self.denotation = SkipGramNegativeSampling(config, data)

        self.decompose = config.decompose_mode
        if self.decompose == 'denotation':
            self.deno_loss_weight = config.𝛿
            self.cono_loss_weight = 0 - config.𝛾
        elif self.decompose == 'connotation':
            self.cono_loss_weight = config.𝛾
            self.deno_loss_weight = 0 - config.𝛿
        else:
            raise ValueError('Unknown decomposition mode')
        print(f'decompose_mode = {self.decompose}')
        self.to(config.device)

    def forward(
            self,
            center_word_ids: Vector,
            context_word_ids: Matrix,
            party_label: Vector
            ) -> Tuple[Scalar, float, float]:
        deno_loss = self.deno_forward(center_word_ids, context_word_ids)
        if self.decompose == 'connotation':
            deno_loss = torch.clamp(deno_loss, self.𝜀, 5)

        cono_logits = self.cono_forward(center_word_ids)
        cono_loss = self.cross_entropy(cono_logits, party_label)
        if self.decompose == 'denotation':
            cono_loss = torch.clamp(cono_loss, self.𝜀, 5)

        loss = (self.deno_loss_weight * deno_loss +
                self.cono_loss_weight * cono_loss)

        loss = torch.clamp(loss, self.𝜀)
        return loss, deno_loss.item(), cono_loss.item()

    def deno_forward(
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

    def cono_forward(self, word_ids: Vector) -> Vector:
        features = self.center_embedding(word_ids)
        logits = self.linear_classifier(self.dropout(features))
        return logits


# TODO move load pretrained embedding to utils
# TODO try torch.chunk, split
class AdversarialDataset(Dataset):

    def __init__(self, config: EncoderConfig):
        self.window_radius = config.window_radius
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

        self.total_num_epoches = config.num_epochs
        self.conf_threshold_schedule = iter(
            np.linspace(0.5, 0.7, num=self.config.num_epochs))

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
            current_epoch: int
            ) -> None:
        conf_threshold = self.conf_threshold_schedule.__next__()

        # predicted_conf = vocab_confidence.max(dim=1)
        predicted_conf: Vector = vocab_confidence.max(dim=1).values
        keep = predicted_conf.gt(conf_threshold)

        subsampled_speeches = []
        for doc in self.speeches:
            subsampled_doc = [word_id for word_id in doc if keep[word_id]]
            if len(subsampled_doc) > config.min_doc_len:
                subsampled_speeches.append(subsampled_doc)
        self.speeches = subsampled_speeches

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

        # self.encoder_optimizer = config.optimizer(model.encoder.parameters())
        self.encoder_optimizer = config.optimizer(
            list(model.center_embedding.parameters()) +
            list(model.encoder.parameters()))
        self.cono_optimizer = config.optimizer(model.linear_classifier.parameters())
        self.deno_optimizer = config.optimizer(
            list(model.denotation_to_center_vec.parameters()) +
            list(model.context_embedding.parameters()))

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
        # x1 = self.model.linear_classifier.weight.grad
        # print(self.model.encoder.weight.grad)
        self.encoder_optimizer.step()

        self.model.zero_grad()
        l_deno.backward(retain_graph=True)
        self.deno_optimizer.step()

        self.model.zero_grad()
        l_cono.backward()
        # x2 = self.model.linear_classifier.weight.grad
        # assert torch.equal(x1, x2)
        self.cono_optimizer.step()

        return l_encoder.item(), l_deno.item(), l_cono.item()

    def fast_train(self, batch: Tuple) -> Tuple[float, float, float]:
        center_word_ids = batch[0].to(self.device)
        context_word_ids = batch[1].to(self.device)
        party_labels = batch[2].to(self.device)

        l_encoder, l_deno, l_cono = self.model(
            center_word_ids, context_word_ids, party_labels)

        self.model.zero_grad()
        # print(self.model.linear_classifier.weight.grad)
        l_encoder.backward()  # computes all gradients
        self.encoder_optimizer.step()  # apply gradients accordingly
        # x = self.model.linear_classifier.weight.grad
        self.deno_optimizer.step()
        # y = self.model.linear_classifier.weight.grad
        self.cono_optimizer.step()
        # z = self.model.linear_classifier.weight.grad
        # assert torch.equal(y, z)
        return l_encoder.item(), l_deno.item(), l_cono.item()

    def train(self) -> None:
        config = self.config
        self.device = config.device

        tb_global_step = 0
        epoch_pbar = tqdm(range(1, config.num_epochs + 1), desc='Epochs')
        for epoch_index in epoch_pbar:
            batch_pbar = tqdm(
                enumerate(self.dataloader), total=len(self.dataloader),
                mininterval=1, desc='Batches')
            for batch_index, batch in batch_pbar:

                l_encoder, l_deno, l_cono = self.safe_train(batch)

                if batch_index % config.update_tensorboard_per_batch == 0:
                    batch_accuracy = self.model.connotation_accuracy(batch)
                    tqdm.write(f'Epoch {epoch_index}, Batch {batch_index:,}:\t'
                               f'𝓁encoder = {l_encoder:.3f}\t'
                               f'({config.denotation_weight}) 𝓁deno = {l_deno:.3f}\t'
                               f'({config.connotation_weight}) 𝓁cono = {l_cono:.3f}\t'
                               f'Batch accuracy = {batch_accuracy:.2%}')

            self.tensorboard.add_scalar(
                'training loss', l_encoder, tb_global_step)
            self.tensorboard.add_scalar(
                'training accuracy', batch_accuracy, tb_global_step)
            self.tensorboard.add_scalar(
                'denotation loss', l_deno, tb_global_step)
            self.tensorboard.add_scalar(
                'connotation loss', l_cono, tb_global_step)
            tb_global_step += 1
            # end batch
            # self.lr_scheduler.step()  # NOTE

            # valid_accuracy = self.model.accuracy(self.data.valid_batch)
            # tqdm.write(f'Epoch {epoch_index}  '
            #            f'Validation Accuracy = {valid_accuracy:.2%}\n\n')
            # self.tensorboard.add_scalar(
            #     'validation accuracy', valid_accuracy, epoch_index)
            if config.export_error_analysis:
                if epoch_index == 1 or epoch_index % 10 == 0:
                    self.model.connotation_accuracy(
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
class DenotationEncoderConfig(ExperimentConfig):
    # Essential
    input_dir: str = '../data/processed/adversarial/44_Obama_1e-5'
    output_dir: str = '../results/adversarial/debug_0'
    device: torch.device = torch.device('cuda:0')
    debug_subset_corpus: Optional[int] = None
    num_dataloader_threads: int = 0

    # Hyperparameters
    # Let 𝛿, 𝛾 ∈ (0, 1)
    denotation_weight: float = 1  # denotation weight
    connotation_weight: float = -1  # connotation weight
    batch_size: int = 8
    embed_size: int = 300
    hidden_size: int = 300
    window_radius: int = 5  # context_size = 2 * window_radius
    num_negative_samples: int = 10
    dropout_p: float = 0
    num_epochs: int = 100

    # Model Construction
    model: nn.Module = DenotationEncoder
    pretrained_embedding: Optional[str] = None
    # pretrained_embedding: Optional[str] = '../results/baseline/word2vec_single.txt'
    freeze_embedding: bool = False
    num_prediction_classes: int = 2
    sparse_embedding: bool = False  # faster if use with sparse optimizer
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

    reload_state_dict_path: Optional[str] = None
    reload_experiment_path: Optional[str] = None
    clear_tensorboard_log_in_output_dir: bool = True
    delete_all_exisiting_files_in_output_dir: bool = False
    auto_save_every_epoch: bool = False
    export_error_analysis: bool = False

    auto_save_before_quit: bool = True
    save_to_tensorboard_embedding_projector: bool = False
    update_tensorboard_per_batch: int = 100
    print_stats_per_batch: int = 100  # TODO this is ignored


@dataclass
class ConnotationEncoderConfig(DenotationEncoderConfig):
    model: nn.Module = ConnotationEncoder


def main() -> None:
    config = DenotationEncoderConfig()
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
