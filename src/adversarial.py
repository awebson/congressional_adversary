import random
import pickle
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Dict, Iterable, Optional
from typing import Counter as CounterType

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from skip_gram import SkipGramNegativeSampling
from utils.experiment import Experiment
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor

random.seed(42)
torch.manual_seed(42)

class AdversarialDecomposer(nn.Module):

    def __init__(self, config: 'AdversarialConfig', data: 'AdversarialDataset'):
        super().__init__()
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.Dem_frequency = data.Dem_frequency
        self.GOP_frequency = data.GOP_frequency
        self.cherry_pick = config.cherry_pick
        self.device = config.device
        vocab_size = len(data.word_to_id)
        embed_size = config.embed_size

        # Initialize Embedding
        if config.pretrained_embedding is not None:
            config.embed_size = data.pretrained_embedding.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                data.pretrained_embedding, sparse=config.sparse_embedding_grad)
        else:
            self.embedding = nn.Embedding(
                vocab_size, embed_size, sparse=config.sparse_embedding_grad)
            init_range = 1.0 / embed_size
            nn.init.uniform_(self.embedding.weight.data,
                             -init_range, init_range)
        self.embedding.weight.requires_grad = not config.freeze_embedding

        # Adversarial Encoder
        self.encoder = nn.Sequential(
            nn.Linear(embed_size, config.encoded_size),
            nn.SELU(),
            # nn.AlphaDropout(p=config.dropout_p),
            # nn.Linear(config.hidden_size, config.encoded_size),
            # nn.SELU()
        )

        self.deno_loss_weight = config.denotation_weight
        self.cono_loss_weight = config.connotation_weight

        # Denotation: Skip-Gram Softmax softmax
        # self.deno_decoder = nn.Linear(config.encoded_size, vocab_size)

        # Dennotation: Skip-Gram Negative Sampling
        self.num_negative_samples = config.num_negative_samples
        SkipGramNegativeSampling.init_negative_sampling(
            self, data.word_frequency, data.word_to_id)

        # Connotation: Party Classifier
        self.cono_decoder = nn.Linear(
            config.encoded_size, config.num_prediction_classes)
        # self.cono_decoder = nn.Sequential(
        #     nn.Linear(config.encoded_size, config.hidden_size),
        #     nn.ReLU(),
        #     # nn.Dropout(p=config.dropout_p),
        #     nn.Linear(config.hidden_size, config.num_prediction_classes),
        # )
        self.cross_entropy = nn.CrossEntropyLoss()
        self.to(self.device)

    def forward(
            self,
            center_word_ids: Vector,
            context_word_ids: Vector,
            party_labels: Vector
            ) -> Tuple[Scalar, Scalar, Scalar]:
        batch_size = len(context_word_ids)
        negative_context_ids = self.negative_sampling_dist.sample(
            (batch_size, self.num_negative_samples))
        encoded_center = self.encoder(self.embedding(center_word_ids))
        encoded_true_context = self.encoder(self.embedding(context_word_ids))
        encoded_negative_context = self.encoder(self.embedding(negative_context_ids))
        deno_loss = self.deno_forward(
            encoded_center, encoded_true_context, encoded_negative_context)

        # encoded_center = self.encoder(self.embedding(center_word_ids))
        # deno_logits = self.deno_decoder(encoded_center)
        # deno_loss = self.cross_entropy(deno_logits, context_word_ids)

        cono_logits = self.cono_decoder(encoded_center)
        cono_loss = self.cross_entropy(cono_logits, party_labels)

        encoder_loss = (self.deno_loss_weight * deno_loss +
                        self.cono_loss_weight * cono_loss)
        return encoder_loss, deno_loss, cono_loss

    def deno_forward(
            self,
            center: Matrix,
            true_context: Matrix,
            negative_context: R3Tensor
            ) -> Scalar:
        """Faster but less readable."""
        # center = self.deno_decoder(encoded_center)
        # # true_context = self.context_decoder(encoded_true_context)
        # # negative_context = self.context_decoder(encoded_negative_context)
        # true_context = self.deno_decoder(encoded_true_context)
        # negative_context = self.deno_decoder(encoded_negative_context)

        # batch_size * embed_size
        objective = torch.sum(
            torch.mul(center, true_context),  # Hadamard product
            dim=1)  # be -> b
        objective = nn.functional.logsigmoid(objective)

        # batch_size * num_negative_examples * embed_size
        # negative_context: bne
        # center: be -> be1
        negative_objective = torch.bmm(  # bne, be1 -> bn1
            negative_context, center.unsqueeze(2)
            ).squeeze()  # bn1 -> bn
        negative_objective = nn.functional.logsigmoid(-negative_objective)
        negative_objective = torch.sum(negative_objective, dim=1)  # bn -> b
        return -torch.mean(objective + negative_objective)

    def predict_connotation(self, word_ids: Vector) -> Vector:
        self.eval()
        with torch.no_grad():
            embed = self.embedding(word_ids)
            encoded = self.encoder(embed)
            logits = self.cono_decoder(encoded)
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

        # Debug High Confidence Predictions
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

        tqdm.write(f'Exporting all vocabulary connnotation to {export_path}')
        output = []
        for conf, word_id in zip(confidence, all_vocab_ids):  # type: ignore
            word = self.id_to_word[word_id.item()]
            Dem_freq = self.Dem_frequency[word]
            GOP_freq = self.GOP_frequency[word]
            output.append((conf.tolist(), Dem_freq, GOP_freq, word))
        output.sort(key=lambda tup: tup[0][0])  # ascending GOP confidence

        if self.cherry_pick:
            cherry_output = []
            for cherry_word in self.cherry_pick:
                try:
                    cherry_conf = confidence[self.word_to_id[cherry_word]]
                except KeyError:
                    continue
                Dem_freq = self.Dem_frequency[cherry_word]
                GOP_freq = self.GOP_frequency[cherry_word]
                cherry_output.append(
                    (cherry_conf.tolist(), Dem_freq, GOP_freq, cherry_word))
            cherry_output.sort(key=lambda tup: tup[0][0])

        # self.accuracy_at_confidence_plot(output)  # TODO

        with open(export_path, 'w') as file:
            file.write('[Dem confidence, GOP confidence]\t'
                       '(Dem frequency, GOP frequency)\n')
            if self.cherry_pick:
                for conf, Dem_freq, GOP_freq, word in cherry_output:
                    file.write(f'[{conf[0]:.2%}, {conf[1]:.2%}]\t\t'
                               f'({Dem_freq}, {GOP_freq})\t\t'
                               f'{word}\n')
                file.write('\n')
            for conf, Dem_freq, GOP_freq, word in output:
                file.write(f'[{conf[0]:.2%}, {conf[1]:.2%}]\t\t'
                           f'({Dem_freq}, {GOP_freq})\t\t'
                           f'{word}\n')

    def export_encoded_embedding(
            self,
            export_path: Optional[str] = None,
            tensorboard: bool = False,
            device: Optional[torch.device] = None
            ) -> Optional[Matrix]:
        """for querying nearest neighbors & visualization"""
        all_vocab_ids = torch.arange(len(self.word_to_id), dtype=torch.long)
        if device:
            all_vocab_ids = all_vocab_ids.to(device)
        else:
            all_vocab_ids = all_vocab_ids.to(self.device)

        self.eval()
        with torch.no_grad():
            encoded_embed = self.encoder(self.embedding(all_vocab_ids))
        self.train()

        if export_path:
            raise NotImplementedError
        if tensorboard:
            embedding_labels = [
                self.id_to_word[word_id]
                for word_id in all_vocab_ids]  # type: ignore
            self.tensorboard.add_embedding(
                encoded_embed, embedding_labels, global_step=0)  # TODO .weight.data?
        return encoded_embed

    def eval_word_similarity(self) -> float:
        from evaluations.external.word_similarity import all_wordsim
        all_vocab_ids = torch.arange(
            len(self.word_to_id), dtype=torch.long, device=self.device)
        frozen_pretrained = self.embedding(all_vocab_ids)
        encoded_embed = self.export_encoded_embedding()
        return all_wordsim.mean_delta(encoded_embed, frozen_pretrained, self.id_to_word)


class PreparsedAdversarialDataset(Dataset):

    def __init__(self, config: 'AdversarialConfig'):
        corpus_path = os.path.join(config.input_dir, 'train_data.pickle')
        with open(corpus_path, 'rb') as corpus_file:
            preprocessed = pickle.load(corpus_file)
        self.word_to_id = preprocessed['word_to_id']
        self.id_to_word = preprocessed['id_to_word']
        self.Dem_frequency: CounterType[str] = preprocessed['Dem_init_freq']
        self.GOP_frequency: CounterType[str] = preprocessed['GOP_init_freq']
        self.word_frequency: CounterType[str] = preprocessed['combined_frequency']
        self.center_word_ids: List[int] = preprocessed['center_word_ids']
        self.context_word_ids: List[int] = preprocessed['context_word_ids']
        self.labels: List[int] = preprocessed['party_labels']
        del preprocessed

        self.pretrained_embedding: Matrix
        if config.pretrained_embedding is not None:
            corpus_id_to_word = self.id_to_word
            self.pretrained_embedding, self.word_to_id, self.id_to_word = (
                Experiment.load_embedding(config.pretrained_embedding))
            self.center_word_ids = Experiment.convert_word_ids(
                self.center_word_ids, corpus_id_to_word, self.word_to_id)
            self.context_word_ids = Experiment.convert_word_ids(
                self.context_word_ids, corpus_id_to_word, self.word_to_id)

        if config.debug_subset_corpus:
            self.center_word_ids = self.center_word_ids[:config.debug_subset_corpus]
            self.context_word_ids = self.context_word_ids[:config.debug_subset_corpus]
            self.labels = self.labels[:config.debug_subset_corpus]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[int, int, int]:
        return (
            self.center_word_ids[index],
            self.context_word_ids[index],
            self.labels[index])


class AdversarialDataset(Dataset):

    def __init__(self, config: 'AdversarialConfig'):
        self.window_radius = config.window_radius
        corpus_path = os.path.join(config.input_dir, 'train_data.pickle')
        with open(corpus_path, 'rb') as corpus_file:
            preprocessed = pickle.load(corpus_file)

        self.word_to_id = preprocessed['word_to_id']
        self.id_to_word = preprocessed['id_to_word']
        self.Dem_frequency: CounterType[str] = preprocessed['D_raw_freq']
        self.GOP_frequency: CounterType[str] = preprocessed['R_raw_freq']
        self.word_frequency: CounterType[str] = preprocessed['combined_frequency']
        self.documents: List[List[int]] = preprocessed['documents']
        self.labels: List[int] = preprocessed['labels']
        del preprocessed

        self.pretrained_embedding: Matrix
        if config.pretrained_embedding is not None:
            self.pretrained_embedding = Experiment.load_embedding(
                config.pretrained_embedding, self.word_to_id)

        if config.debug_subset_corpus:
            self.documents = self.documents[:config.debug_subset_corpus]
            self.labels = self.labels[:config.debug_subset_corpus]


    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Vector, Vector, Vector]:
        """
        parse one document into a List[skip-grams], where each skip-gram
        is a Tuple[center_id, List[context_ids]]
        """
        doc: List[int] = self.documents[index]
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
                (len(context_word_ids),), self.labels[index], dtype=torch.int64))

    @staticmethod
    def collate(
            faux_batch: List[Tuple[Vector, Vector, Vector]]
            ) -> Tuple[Vector, Vector, Vector]:
        center_word_ids = torch.cat([center for center, _, _ in faux_batch])
        context_word_ids = torch.cat([context for _, context, _ in faux_batch])
        labels = torch.cat([label for _, _, label in faux_batch])
        # shuffle = torch.randperm(len(labels))
        # return center_word_ids[shuffle], context_word_ids[shuffle], labels[shuffle]
        return center_word_ids, context_word_ids, labels


class AdversarialExperiment(Experiment):

    def __init__(self, config: 'AdversarialConfig'):
        super().__init__(config)
        self.data = AdversarialDataset(config)
        self.dataloader = DataLoader(
            self.data,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.data.collate,
            num_workers=config.num_dataloader_threads)
        self.model = AdversarialDecomposer(config, self.data)

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)  # param.data)

        if config.freeze_embedding:
            self.encoder_optimizer = config.optimizer(
                self.model.encoder.parameters(),
                lr=config.learning_rate)
        else:
            self.encoder_optimizer = config.optimizer(
                list(self.model.embedding.parameters()) +
                list(self.model.encoder.parameters()),
                lr=config.learning_rate)

        self.cono_optimizer = config.optimizer(
            self.model.cono_decoder.parameters(),
            lr=config.learning_rate)
        # self.deno_optimizer = config.optimizer(
        #     self.model.deno_decoder.parameters(),
        #     lr=config.learning_rate)

        self.to_be_saved = {
            'config': self.config,
            'model': self.model,
            'encoder_optimizer': self.encoder_optimizer,
            # 'deno_optimizer': self.deno_optimizer,
            'cono_optimizer': self.cono_optimizer}

        self.custom_stats_format = (
            '𝓁encoder = {Loss/encoder:.3f}\t'
            f'({config.denotation_weight}) '
            '𝓁deno = {Loss/denotation:.3f}\t'
            f'({config.connotation_weight}) '
            '𝓁cono = {Loss/connotation:.3f}\t'
            'cono accuracy = {Evaluation/connotation_accuracy:.2%}')

    def _train(
            self,
            center_word_ids: Vector,
            context_word_ids: Vector,
            party_labels: Vector
            ) -> Tuple[float, float, float]:
        self.model.zero_grad()
        l_encoder, l_deno, l_cono = self.model(
            center_word_ids, context_word_ids, party_labels)
        l_encoder.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.encoder_optimizer.step()

        # self.model.zero_grad()
        # l_deno.backward(retain_graph=True)
        # nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        # self.deno_optimizer.step()

        self.model.zero_grad()
        l_cono.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.cono_optimizer.step()
        return l_encoder.item(), l_deno.item(), l_cono.item()

    def fast_train(
            self,
            center_word_ids: Vector,
            context_word_ids: Vector,
            party_labels: Vector
            ) -> Tuple[float, float, float]:
        l_encoder, l_deno, l_cono = self.model(
            center_word_ids, context_word_ids, party_labels)
        self.model.zero_grad()
        l_encoder.backward()  # computes all gradients
        self.encoder_optimizer.step()  # apply gradients accordingly
        # self.encoder_optimizer.zero_grad()
        # (re)divide encoder loss by gamma?
        self.deno_optimizer.step()
        # self.deno_optimizer.zero_grad()
        self.cono_optimizer.step()
        return l_encoder.item(), l_deno.item(), l_cono.item()

    def train(self) -> None:
        config = self.config
        epoch_pbar = tqdm(range(1, config.num_epochs + 1), desc='Epochs')
        for epoch_index in epoch_pbar:
            batch_pbar = tqdm(
                enumerate(self.dataloader), total=len(self.dataloader),
                mininterval=1, desc='Batches')
            for batch_index, batch in batch_pbar:
                center_word_ids = batch[0].to(self.device)
                context_word_ids = batch[1].to(self.device)
                party_labels = batch[2].to(self.device)

                l_encoder, l_deno, l_cono = self._train(
                    center_word_ids, context_word_ids, party_labels)

                if batch_index % config.update_tensorboard == 0:
                    cono_accuracy = self.model.connotation_accuracy(batch)
                    stats = {
                        'Loss/encoder': l_encoder,
                        'Loss/denotation': l_deno,
                        'Loss/connotation': l_cono,
                        'Evaluation/connotation_accuracy': cono_accuracy
                    }
                    self.update_tensorboard(stats)
                if batch_index % config.print_stats == 0:
                    self.print_stats(epoch_index, batch_index, stats)
                if batch_index % 10_000 == 0:
                    mean_delta = self.model.eval_word_similarity()
                    self.tensorboard.add_scalar(
                        'Evaluation/word_sim_mean_delta', mean_delta, self.tb_global_step)

            # End Batches
            # self.lr_scheduler.step()
            self.print_timestamp()
            self.auto_save(epoch_index)
            if config.export_error_analysis:
                if (epoch_index % config.export_error_analysis == 0
                        or epoch_index == 1):
                    export_path = os.path.join(
                        config.output_dir,
                        f'vocab_cono_epoch{epoch_index}.txt')
                    self.model.all_vocab_connotation(export_path)
                    # mean_delta = self.model.eval_word_similarity()
                    # self.tensorboard.add_scalar(
                    #     'word_sim_mean_delta', mean_delta, epoch_index)

            # Discard partisan neutral words
            # if config.subsampling_threshold_schedule:
            #     vocab_party_confidence = self.model.all_vocab_connotation()
            #     subsampled_num_words = self.data.subsample_neutral_words(
            #         vocab_party_confidence)
            #     if subsampled_num_words:
            #         self.update_tensorboard(
            #             {'Accuracy/num_words_subsampled_corpus':
            #                 subsampled_num_words})
        # End Epochs


@dataclass
class AdversarialConfig():
    # Essential
    input_dir: str = '../data/processed/adversarial/1e-5/Obama'
    output_dir: str = '../results/adversarial/debug'
    device: torch.device = torch.device('cuda')
    debug_subset_corpus: Optional[int] = None
    num_dataloader_threads: int = 0

    # Hyperparameters
    denotation_weight: float = 1  # denotation weight 𝛿
    connotation_weight: float = -1  # connotation weight 𝛾
    batch_size: int = 8
    embed_size: int = 300
    hidden_size: int = 300  # MLP encoder
    encoded_size: int = 300  # encoder output
    window_radius: int = 5  # context_size = 2 * window_radius
    num_negative_samples: int = 10
    dropout_p: float = 0
    num_epochs: int = 10
    pretrained_embedding: Optional[str] = None
    freeze_embedding: bool = True
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    learning_rate: float = 1e-4
    # lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    num_prediction_classes: int = 2
    sparse_embedding_grad: bool = False  # faster if used with sparse optimizer
    init_trick: bool = False
    # 𝜀: float = 1e-5

    # # Subsampling Trick
    # subsampling_threshold_schedule: Optional[Iterable[float]] = None
    # subsampling_power: float = 8  # float('inf') for deterministic sampling
    # min_doc_length: int = 2  # for subsampling partisan neutral words

    # Evaluation
    cherry_pick: Optional[Tuple[str, ...]] = (
        'estate_tax', 'death_tax',
        'american_clean_energy', 'national_energy_tax',
        'undocumented', 'immigrants', 'illegal_immigrants', 'illegal_aliens',
        'protection_and_affordable', 'the_affordable_care_act',
        'obamacare', 'health_care_bill', 'socialized_medicine', 'public_option',
        'financial_stability',
        'capital_gains_tax',
        'second_amendment_rights',
        'government_spending',
        'offensive_missiles', 'star_wars_program', 'icbms',
        'corporate_profits', 'earnings',
        'retroactive_immunity', 'the_fisa_bill'
    )

    # Housekeeping
    export_error_analysis: Optional[int] = 1  # per epoch
    update_tensorboard: int = 1000  # per batch
    print_stats: int = 1000  # per batch
    reload_path: Optional[str] = None
    clear_tensorboard_log_in_output_dir: bool = True
    delete_all_exisiting_files_in_output_dir: bool = False
    auto_save_per_epoch: Optional[int] = 5
    auto_save_if_interrupted: bool = False
    export_tensorboard_embedding_projector: bool = False


def main() -> None:
    today = datetime.now().strftime("%m-%d %a")

    # # Debug
    # config = AdversarialConfig(
    #     input_dir='../data/processed/labeled_speeches/1e-5/for_real',
    #     output_dir='../results/debug',
    #     # debug_subset_corpus=1000,
    #     denotation_weight=1,
    #     connotation_weight=0,
    #     pretrained_embedding='../data/pretrained_word2vec/for_real.txt',
    #     # init_trick=True,
    #     encoded_size=300,
    #     # hidden_size=300,
    #     num_epochs=10,
    #     batch_size=4,
    #     auto_save_per_epoch=1,
    #     device=torch.device('cuda:0')
    # )


    # # Vanilla denotation
    # config = AdversarialConfig(
    #     input_dir='../data/processed/labeled_speeches/1e-5/for_real',
    #     output_dir='../results/adversarial/for_real_NS/1d 0c',
    #     denotation_weight=1,
    #     connotation_weight=0,
    #     pretrained_embedding='../data/pretrained_word2vec/for_real.txt',
    #     # init_trick=True,
    #     encoded_size=300,
    #     # hidden_size=300,
    #     num_epochs=30,
    #     batch_size=2,
    #     auto_save_per_epoch=3,
    #     num_dataloader_threads=6,
    #     device=torch.device('cuda:1')
    # )

    # # Deno minus cono 1d - 10c
    config = AdversarialConfig(
        input_dir='../data/processed/labeled_speeches/1e-5/for_real',
        output_dir='../results/adversarial/for_real_NS/1d -10c L2',
        denotation_weight=1,
        connotation_weight=-10,
        pretrained_embedding='../data/pretrained_word2vec/for_real.txt',
        encoded_size=300,
        hidden_size=300,
        # dropout_p=0.1,
        num_epochs=30,
        batch_size=6,
        auto_save_per_epoch=3,
        device=torch.device('cuda:0')
    )

    # config = AdversarialConfig(
    #     input_dir='../data/processed/labeled_speeches/1e-5/for_real',
    #     output_dir='../results/adversarial/for_real_NS/1d -12c L2',
    #     denotation_weight=1,
    #     connotation_weight=-12,
    #     pretrained_embedding='../data/pretrained_word2vec/for_real.txt',
    #     encoded_size=300,
    #     hidden_size=300,
    #     # dropout_p=0.1,
    #     num_epochs=30,
    #     batch_size=3,
    #     auto_save_per_epoch=3,
    #     device=torch.device('cuda:1')
    # )

    # # Vanilla connotation
    # config = AdversarialConfig(
    #     input_dir='../data/processed/labeled_speeches/1e-5/W_Bush',
    #     output_dir=f'../results/W_Bush/0d 1c',
    #     denotation_weight=0,
    #     connotation_weight=1,
    #     pretrained_embedding='../data/pretrained_word2vec/W_Bush.txt',
    #     encoded_size=300,
    #     num_epochs=30,
    #     batch_size=4096,
    #     auto_save_per_epoch=3,
    #     num_dataloader_threads=4,
    #     device=torch.device('cuda:1')
    # )

    # # Cono minus deno -0.05d + 1c
    # config = AdversarialConfig(
    #     output_dir=f'../results/adversarial/Obama/-0.001d1c NTi',
    #     denotation_weight=-0.001,
    #     connotation_weight=1,
    #     pretrained_embedding='../data/pretrained_word2vec/word2vec_Obama.txt',
    #     init_trick=False,
    #     encoded_size=300,
    #     num_epochs=30,
    #     batch_size=1024,
    #     auto_save_per_epoch=3,
    #     device=torch.device('cuda:1')
    # )

    black_box = AdversarialExperiment(config)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    main()
