import argparse
import random
import pickle
import os
from dataclasses import dataclass
from typing import Tuple, List, Dict, Iterable, Optional
from typing import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from skip_gram import SkipGramNegativeSampling
from utils.experiment import Experiment
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor
from utils.word_similarity import all_wordsim
from evaluations import intrinsic_eval

random.seed(42)
torch.manual_seed(42)


class Decomposer(nn.Module):

    def __init__(
            self,
            architecture: nn.Module,
            decomposed_size: int,
            delta: float,
            gamma: float,
            embed_size: int,
            num_prediction_classes: int,
            device: torch.device):
        super().__init__()

        # Decomposer
        self.encoder = architecture
        self.delta = delta
        self.gamma = gamma
        self.device = device

        # Dennotation Loss: Skip-Gram Negative Sampling
        self.deno_decoder = nn.Linear(decomposed_size, embed_size)

        # Connotation Loss: Party Classifier
        self.cono_decoder = nn.Linear(decomposed_size, num_prediction_classes)
        self.to(self.device)

    def forward(
            self,
            center: Vector,
            true_context: Vector,
            negative_context: Vector,
            party_labels: Vector
            ) -> Tuple[Vector, Scalar, Scalar, Scalar]:
        encoded_center = self.encoder(center)
        encoded_true_context = self.encoder(true_context)
        encoded_negative_context = self.encoder(negative_context)

        deno_loss = self.deno_forward(
            encoded_center, encoded_true_context, encoded_negative_context)

        cono_logits = self.cono_decoder(encoded_center)
        cono_loss = nn.functional.cross_entropy(cono_logits, party_labels)

        decomposer_loss = (self.delta * deno_loss +
                           self.gamma * cono_loss)
        return encoded_center, decomposer_loss, deno_loss, cono_loss

    def deno_forward(
            self,
            encoded_center: Matrix,
            encoded_true_context: Matrix,
            encoded_negative_context: R3Tensor
            ) -> Scalar:
        """Faster but less readable."""
        center = self.deno_decoder(encoded_center)
        true_context = self.deno_decoder(encoded_true_context)
        negative_context = self.deno_decoder(encoded_negative_context)

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


class Recomposer(nn.Module):

    def __init__(self, config: 'RecomposerConfig', data: 'SpeechesWithPartyDict'):
        super().__init__()
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.Dem_frequency = data.Dem_frequency
        self.GOP_frequency = data.GOP_frequency
        self.cherries = config.cherries
        self.device = config.device
        vocab_size = len(data.word_to_id)
        # embed_size = config.embed_size

        config.embed_size = data.pretrained_embedding.shape[1]
        self.pretrained_embed = nn.Embedding.from_pretrained(data.pretrained_embedding)
        self.pretrained_embed.weight.requires_grad = not config.freeze_embedding

        self.num_negative_samples = config.num_negative_samples
        SkipGramNegativeSampling.init_negative_sampling(
            self, data.subsampled_frequency, data.word_to_id)

        # Decomposers
        self.decomposer_f = Decomposer(
            config.deno_architecture,
            config.decomposed_deno_size,
            config.deno_delta,
            config.deno_gamma,
            config.embed_size,
            config.num_prediction_classes,
            config.device)
        self.decomposer_g = Decomposer(
            config.cono_architecture,
            config.decomposed_cono_size,
            config.cono_delta,
            config.cono_gamma,
            config.embed_size,
            config.num_prediction_classes,
            config.device)

        # Recomposer (Autoencoder)
        self.recomposer_h = nn.Linear(
            config.decomposed_deno_size + config.decomposed_cono_size,
            config.embed_size)
        self.rho = config.recomposer_rho
        self.to(self.device)

    def forward(
            self,
            center_word_ids: Vector,
            context_word_ids: Vector,
            party_labels: Vector
            ) -> Tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]:
        batch_size = context_word_ids.shape[0]
        negative_context_ids = torch.multinomial(
            self.categorical_dist_probs,
            batch_size * self.num_negative_samples,
            replacement=True,
        ).view(batch_size, self.num_negative_samples).to(self.device)

        center = self.pretrained_embed(center_word_ids)
        true_context = self.pretrained_embed(context_word_ids)
        negative_context = self.pretrained_embed(negative_context_ids)

        V_deno, L_f, l_f_deno, l_f_cono = self.decomposer_f(
            center, true_context, negative_context, party_labels)
        V_cono, L_g, l_g_deno, l_g_cono = self.decomposer_g(
            center, true_context, negative_context, party_labels)

        recomposed = self.recomposer_h(
            torch.cat((V_deno, V_cono), dim=1))
        l_h: Scalar = 1 - torch.mean(  # type: ignore
            nn.functional.cosine_similarity(center, recomposed))
        L_master = L_f + L_g + self.rho * l_h
        return L_master, l_f_deno, l_f_cono, l_g_deno, l_g_cono, l_h

    def export_embeddings(
            self, device: Optional[torch.device] = None
            ) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
        """for querying nearest neighbors & visualization"""
        all_vocab_ids = torch.arange(
            self.pretrained_embed.num_embeddings, dtype=torch.long)
        if not device:
            device = self.device
        all_vocab_ids = all_vocab_ids.to(device)

        self.eval()
        with torch.no_grad():
            pretrained = self.pretrained_embed(all_vocab_ids)
            deno_embed = self.decomposer_f.encoder(pretrained)
            cono_embed = self.decomposer_g.encoder(pretrained)
            recomp_embed = self.recomposer_h(
                torch.cat((deno_embed, cono_embed), dim=1))
        self.train()
        return pretrained, deno_embed, cono_embed, recomp_embed

    def predict_connotation(self, word_ids: Vector) -> Vector:
        self.eval()
        with torch.no_grad():
            embed = self.pretrained_embed(word_ids)

            deno = self.decomposer_f.encoder(embed)
            cono = self.decomposer_g.encoder(embed)

            f_logits = self.decomposer_f.cono_decoder(deno)
            g_logits = self.decomposer_g.cono_decoder(cono)

            f_confidence = nn.functional.softmax(f_logits, dim=1)
            g_confidence = nn.functional.softmax(g_logits, dim=1)
        self.train()
        return f_confidence, g_confidence

    def connotation_accuracy(
            self,
            word_ids: Vector,
            labels: Vector,
            export_error_analysis: Optional[str] = None
            ) -> Tuple[float, float]:
        f_conf, g_conf = self.predict_connotation(word_ids)
        f_predictions = f_conf.argmax(dim=1)
        g_predictions = g_conf.argmax(dim=1)
        f_correct_indicies = f_predictions.eq(labels)
        g_correct_indicies = g_predictions.eq(labels)
        f_accuracy = f_correct_indicies.float().mean().item()
        g_accuracy = g_correct_indicies.float().mean().item()
        return f_accuracy, g_accuracy

    def all_vocab_connotation(
            self,
            export_path: Optional[str] = None
            ) -> None:
        """Inspect the decomposed vectors"""
        all_vocab_ids = torch.arange(
            len(self.word_to_id), dtype=torch.long, device=self.device)
        f_confidence, g_confidence = self.predict_connotation(all_vocab_ids)
        # if not export_path:
        #     return confidence

        tqdm.write(f'Exporting all vocabulary connnotation to {export_path}')
        output = []
        for f_conf, g_conf, word_id in zip(  # type: ignore
                f_confidence, g_confidence, all_vocab_ids):
            word = self.id_to_word[word_id.item()]
            Dem_freq = self.Dem_frequency[word]
            GOP_freq = self.GOP_frequency[word]
            output.append(
                (f_conf.tolist(), g_conf.tolist(), Dem_freq, GOP_freq, word))
        output.sort(key=lambda tup: tup[1][0])  # ascending GOP confidence

        if self.cherries:
            cherry_output = []
            for cherry_word in self.cherries:
                try:
                    cherry_f_conf = f_confidence[self.word_to_id[cherry_word]]
                    cherry_g_conf = g_confidence[self.word_to_id[cherry_word]]
                except KeyError:
                    continue
                Dem_freq = self.Dem_frequency[cherry_word]
                GOP_freq = self.GOP_frequency[cherry_word]
                cherry_output.append(
                    (cherry_f_conf.tolist(), cherry_g_conf, Dem_freq, GOP_freq, cherry_word))
            cherry_output.sort(key=lambda tup: tup[1][0])

        def pretty_format(conf):
            out = [f'{c:.2%}' for c in conf]
            out = ', '.join(out)
            return '[' + out + ']'

        # self.accuracy_at_confidence_plot(output)  # TODO
        with open(export_path, 'w') as file:  # type: ignore
            file.write('Deno_space_conf\t\tCono_space_conf\t\t'
                       '(Dem frequency, GOP frequency)\n')
            if self.cherries:
                for f_conf, g_conf, Dem_freq, GOP_freq, word in cherry_output:
                    file.write(f'{pretty_format(f_conf)}\t'
                               f'{pretty_format(g_conf)}\t'
                               f'({Dem_freq}, {GOP_freq})\t'
                               f'{word}\n')
                file.write('\n')
            for f_conf, g_conf, Dem_freq, GOP_freq, word in output:
                file.write(f'{pretty_format(f_conf)}\t'
                           f'{pretty_format(g_conf)}\t'
                           f'({Dem_freq}, {GOP_freq})\t'
                           f'{word}\n')


class SpeechesWithPartyDict(Dataset):

    def __init__(self, config: 'RecomposerConfig'):
        corpus_path = os.path.join(config.input_dir, 'train_data.pickle')
        self.window_radius = config.window_radius
        with open(corpus_path, 'rb') as corpus_file:
            preprocessed = pickle.load(corpus_file)

        self.word_to_id = preprocessed['word_to_id']
        self.id_to_word = preprocessed['id_to_word']
        self.Dem_frequency: Counter[str] = preprocessed['D_raw_freq']
        self.GOP_frequency: Counter[str] = preprocessed['R_raw_freq']
        self.id_to_label: Dict[int, int] = preprocessed['id_to_label']
        self.subsampled_frequency: Counter[str] = preprocessed['subsampled_frequency']
        self.documents: List[List[int]] = preprocessed['speeches']
        del preprocessed
        if config.export_tensorboard_embedding_projector:
            self.vocabulary = [
                self.id_to_word[word_id]
                for word_id in range(len(self.word_to_id))]

        # self.train_word_ids = word_ids[config.dev_holdout:]
        # self.train_labels = labels[config.dev_holdout:]
        # self.dev_word_ids = torch.tensor(word_ids[:config.dev_holdout])
        # self.dev_labels = torch.tensor(labels[:config.dev_holdout])

        if config.pretrained_embedding is not None:
            self.pretrained_embedding: Matrix = Experiment.load_embedding(
                config.pretrained_embedding, self.word_to_id)
        if config.debug_subset_corpus:
            self.train_word_ids = self.train_word_ids[:config.debug_subset_corpus]
            self.train_labels = self.train_labels[:config.debug_subset_corpus]

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> Tuple[Vector, Vector, Vector]:
        """
        parse one document into a List[skip-grams], where each skip-gram
        is a Tuple[center_id, List[context_ids]]
        """
        doc: List[int] = self.documents[index]
        center_word_ids: List[int] = []
        context_word_ids: List[int] = []
        labels: List[int] = []
        for center_index, center_word_id in enumerate(doc):

            left_index = max(center_index - self.window_radius, 0)
            right_index = min(center_index + self.window_radius, len(doc) - 1)
            context_word_id: List[int] = (
                doc[left_index:center_index] +
                doc[center_index + 1:right_index + 1])
            context_word_ids += context_word_id
            center_word_ids += [center_word_id] * len(context_word_id)
            labels += [self.id_to_label[center_word_id]] * len(context_word_id)
        return (
            torch.tensor(center_word_ids),
            torch.tensor(context_word_ids),
            torch.tensor(labels))

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


class RecomposerExperiment(Experiment):

    def __init__(self, config: 'RecomposerConfig'):
        super().__init__(config)
        self.data = SpeechesWithPartyDict(config)
        self.dataloader = DataLoader(
            self.data,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.data.collate,
            num_workers=config.num_dataloader_threads,
            pin_memory=config.pin_memory)
        self.model = Recomposer(config, self.data)
        self.load_cherries()

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)  # param.data)

        self.f_deno_optimizer = config.optimizer(
            self.model.decomposer_f.deno_decoder.parameters(),
            lr=config.learning_rate)
        self.f_cono_optimizer = config.optimizer(
            self.model.decomposer_f.cono_decoder.parameters(),
            lr=config.learning_rate)

        self.g_deno_optimizer = config.optimizer(
            self.model.decomposer_g.deno_decoder.parameters(),
            lr=config.learning_rate)
        self.g_cono_optimizer = config.optimizer(
            self.model.decomposer_g.cono_decoder.parameters(),
            lr=config.learning_rate)

        self.recomposer_optimizer = config.optimizer(
            self.model.decomposer_g.cono_decoder.parameters(),
            lr=config.learning_rate)
        self.decomposer_optimizer = config.optimizer((
            list(self.model.decomposer_f.encoder.parameters()) +
            list(self.model.decomposer_g.encoder.parameters())),
            lr=config.learning_rate)

        self.to_be_saved = {
            'config': self.config,
            'model': self.model}

        self.custom_stats_format = (
            'ℒ = {Recomposer/combined_loss:.3f}\t'
            'Dd = {Denotation Decomposer/deno:.3f}\t'
            'Dc = {Denotation Decomposer/cono:.3f}\t'
            'Cd = {Connotation Decomposer/deno:.3f}\t'
            'Cc = {Connotation Decomposer/cono:.3f}\t'
            'R = {Recomposer/recomp:.3f}\t'
            # 'cono accuracy = {Evaluation/connotation_accuracy:.2%}'
        )

    def _train(
            self,
            word_ids: Vector,
            context_ids: Vector,
            party_labels: Vector,
            grad_clip: float = 5
            ) -> Tuple[float, float, float, float, float, float]:
        self.model.zero_grad()
        L_master, l_f_deno, l_f_cono, l_g_deno, l_g_cono, l_h = self.model(
            word_ids, context_ids, party_labels)
        L_master.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model.decomposer_f.encoder.parameters(), grad_clip)
        nn.utils.clip_grad_norm_(self.model.decomposer_g.encoder.parameters(), grad_clip)
        self.decomposer_optimizer.step()

        self.model.zero_grad()
        l_f_deno.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model.decomposer_f.deno_decoder.parameters(), grad_clip)
        self.f_deno_optimizer.step()

        self.model.zero_grad()
        l_f_cono.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model.decomposer_f.cono_decoder.parameters(), grad_clip)
        self.f_cono_optimizer.step()

        self.model.zero_grad()
        l_g_deno.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model.decomposer_g.deno_decoder.parameters(), grad_clip)
        self.g_deno_optimizer.step()

        self.model.zero_grad()
        l_g_cono.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model.decomposer_g.cono_decoder.parameters(), grad_clip)
        self.g_cono_optimizer.step()

        self.model.zero_grad()
        l_h.backward()
        nn.utils.clip_grad_norm_(self.model.decomposer_g.cono_decoder.parameters(), grad_clip)
        self.recomposer_optimizer.step()

        return (
            L_master.item(),
            l_f_deno.item(),
            l_f_cono.item(),
            l_g_deno.item(),
            l_g_cono.item(),
            l_h.item())

    def train(self) -> None:
        config = self.config
        epoch_pbar = tqdm(range(1, config.num_epochs + 1), desc='Epochs')
        for epoch_index in epoch_pbar:
            batch_pbar = tqdm(
                enumerate(self.dataloader), total=len(self.dataloader),
                mininterval=config.progress_bar_refresh_rate, desc='Batches')
            for batch_index, batch in batch_pbar:
                word_ids = batch[0].to(self.device)
                context_ids = batch[1].to(self.device)
                party_labels = batch[2].to(self.device)

                L_master, l_f_deno, l_f_cono, l_g_deno, l_g_cono, l_h = self._train(
                    word_ids, context_ids, party_labels)

                if batch_index % config.update_tensorboard == 0:
                    f_accuracy, g_accuracy = self.model.connotation_accuracy(
                        word_ids, party_labels)
                    stats = {
                        'Denotation Decomposer/deno': l_f_deno,
                        'Denotation Decomposer/cono': l_f_cono,
                        'Denotation Decomposer/accuracy_train': f_accuracy,
                        'Connotation Decomposer/deno': l_g_deno,
                        'Connotation Decomposer/cono': l_g_cono,
                        'Connotation Decomposer/accuracy_train': g_accuracy,
                        'Recomposer/recomp': l_h,
                        'Recomposer/combined_loss': L_master
                    }
                    self.update_tensorboard(stats)
                if batch_index % config.print_stats == 0:
                    self.print_stats(epoch_index, batch_index, stats)
                if batch_index % config.eval_dev_set == 0:
                    # f_accuracy, g_accuracy = self.model.connotation_accuracy(
                    #     self.data.dev_word_ids.to(self.device),
                    #     self.data.dev_labels.to(self.device))
                    self.cherry_pick()
                    V_pretrained, V_deno, V_cono, V_recomp = self.model.export_embeddings()
                    deno_check = all_wordsim.mean_delta(V_deno, V_pretrained, self.data.id_to_word)
                    cono_check = all_wordsim.mean_delta(V_cono, V_pretrained, self.data.id_to_word)
                    recomp_check = all_wordsim.mean_delta(V_recomp, V_pretrained, self.data.id_to_word)
                    self.update_tensorboard({
                        'Denotation Decomposer/nonpolitical_word_sim_cf_pretrained': deno_check,
                        'Denotation Decomposer/accuracy_dev': f_accuracy,
                        'Connotation Decomposer/nonpolitical_word_sim_cf_pretrained': cono_check,
                        'Connotation Decomposer/accuracy_dev': g_accuracy,
                        'Recomposer/nonpolitical_word_sim_cf_pretrained': recomp_check})
                self.tb_global_step += 1
            # End Batches
            # self.lr_scheduler.step()
            self.print_timestamp()
            self.auto_save(epoch_index)

            if config.export_error_analysis:
                if (epoch_index % config.export_error_analysis == 0
                        or epoch_index == 1):
                    self.model.all_vocab_connotation(os.path.join(
                        config.output_dir, f'vocab_cono_epoch{epoch_index}.txt'))

            if config.export_tensorboard_embedding_projector:
                V_pretrained, V_deno, V_cono, V_recomp = self.model.export_embeddings()
                self.tensorboard.add_embedding(
                    V_deno, self.data.vocabulary, global_step=epoch_index, tag='deno_embed')
                self.tensorboard.add_embedding(
                    V_cono, self.data.vocabulary, global_step=epoch_index, tag='cono_embed')
        # End Epochs

    def load_cherries(self) -> None:
        Dem_pairs = intrinsic_eval.load_cherry(
            '../data/evaluation/cherries/labeled_Dem_samples.tsv',
            exclude_hard_examples=True)
        GOP_pairs = intrinsic_eval.load_cherry(
            '../data/evaluation/cherries/labeled_GOP_samples.tsv',
            exclude_hard_examples=True)
        val_data = Dem_pairs + GOP_pairs

        euphemism = list(
            filter(intrinsic_eval.is_euphemism, val_data))
        party_platform = list(
            filter(intrinsic_eval.is_party_platform, val_data))
        party_platform += intrinsic_eval.load_cherry(
            '../data/evaluation/cherries/remove_deno.tsv',
            exclude_hard_examples=False)

        def get_pretrained_embed(eval_set):
            query_ids = []
            neighbor_ids = []
            for pair in eval_set:
                query_ids.append(self.data.word_to_id[pair.query])
                neighbor_ids.append(self.data.word_to_id[pair.neighbor])
            query = self.model.pretrained_embed(
                torch.tensor(query_ids).to(self.device))
            neighbor = self.model.pretrained_embed(
                torch.tensor(neighbor_ids).to(self.device))
            return query, neighbor

        self.euphemism = get_pretrained_embed(euphemism)
        self.party_platform = get_pretrained_embed(party_platform)

    def cherry_pick(self) -> None:
        eu_q, eu_n = self.euphemism
        pp_q, pp_n = self.party_platform
        sim_eu_pretrained = nn.functional.cosine_similarity(eu_q, eu_n)
        sim_pp_pretrained = nn.functional.cosine_similarity(pp_q, pp_n)
        self.model.eval()
        with torch.no_grad():
            eu_q_deno = self.model.decomposer_f.encoder(eu_q)
            eu_n_deno = self.model.decomposer_f.encoder(eu_n)
            eu_q_cono = self.model.decomposer_g.encoder(eu_q)
            eu_n_cono = self.model.decomposer_g.encoder(eu_n)
            pp_q_deno = self.model.decomposer_f.encoder(pp_q)
            pp_n_deno = self.model.decomposer_f.encoder(pp_n)
            pp_q_cono = self.model.decomposer_g.encoder(pp_q)
            pp_n_cono = self.model.decomposer_g.encoder(pp_n)
        self.model.train()

        sim_eu_deno = nn.functional.cosine_similarity(eu_q_deno, eu_n_deno)
        sim_eu_cono = nn.functional.cosine_similarity(eu_q_cono, eu_n_cono)
        sim_pp_deno = nn.functional.cosine_similarity(pp_q_deno, pp_n_deno)
        sim_pp_cono = nn.functional.cosine_similarity(pp_q_cono, pp_n_cono)

        diff_eu_deno = torch.mean(sim_eu_deno - sim_eu_pretrained)
        diff_eu_cono = torch.mean(sim_eu_cono - sim_eu_pretrained)
        diff_pp_deno = torch.mean(sim_pp_deno - sim_pp_pretrained)
        diff_pp_cono = torch.mean(sim_pp_cono - sim_pp_pretrained)

        self.update_tensorboard({
            'Denotation Decomposer/euphemism_cf_pretrained': diff_eu_deno,
            'Denotation Decomposer/party_platform_cf_pretrained': diff_pp_deno,
            'Denotation Decomposer/euphemism_delta_minus_party_delta': diff_eu_deno - diff_pp_deno,

            'Connotation Decomposer/euphemism_cf_pretrained': diff_eu_cono,
            'Connotation Decomposer/party_platform_cf_pretrained': diff_pp_cono,
            'Connotation Decomposer/euphemism_delta_minus_party_delta': diff_eu_cono - diff_pp_cono
        })

    # def correlate_sim_deltas(self) -> float:
    #     label_deltas = []
    #     model_deltas = []

    #     for pair in self.phrase_pairs:
    #         try:
    #             sim = model.cosine_similarity(pair.query, pair.neighbor)
    #             ref_sim = ref_model.cosine_similarity(pair.query, pair.neighbor)
    #         except KeyError:
    #             continue
    #         model_delta = sim - ref_sim
    #         model_deltas.append(model_delta)
    #         label_deltas.append(pair.deno_sim - pair.cono_sim)

    #         if verbose:
    #             print(f'{pair.deno_sim}  {pair.cono_sim}  {ref_sim:.2%}  {sim:.2%}  '
    #                 f'{pair.query}  {pair.neighbor}')

    #     median = np.median(model_deltas)
    #     mean = np.mean(model_deltas)
    #     stddev = np.std(model_deltas)
    #     rho, _ = spearmanr(model_deltas, label_deltas)
    #     return rho, median, mean, stddev


@dataclass
class RecomposerConfig():
    # Essential
    input_dir: str = '../data/processed/bucket_labeled_speeches/1e-5/for_real'
    output_dir: str = '../results/debug'
    device: torch.device = torch.device('cuda')
    debug_subset_corpus: Optional[int] = None
    num_dataloader_threads: int = 0
    dev_holdout: int = 10_000
    test_holdout: int = 10_000
    pin_memory: bool = False

    # Denotation Decomposer
    decomposed_deno_size: int = 300
    deno_delta: float = 1  # denotation weight 𝛿
    deno_gamma: float = -15  # connotation weight 𝛾

    # Conotation Decomposer
    decomposed_cono_size: int = 300
    cono_delta: float = -0.02  # denotation weight 𝛿
    cono_gamma: float = 1  # connotation weight 𝛾

    # Recomposer
    recomposer_rho: float = 1
    # dropout_p: float = 0.1

    batch_size: int = 2
    embed_size: int = 300
    num_epochs: int = 30
    pretrained_embedding: Optional[str] = '../data/pretrained_word2vec/for_real.txt'
    freeze_embedding: bool = True
    window_radius: int = 5
    num_negative_samples: int = 10
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    # optimizer: torch.optim.Optimizer = torch.optim.SGD
    learning_rate: float = 1e-4
    # momentum: float = 0.5
    # lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    num_prediction_classes: int = 5
    sparse_embedding_grad: bool = False  # faster if used with sparse optimizer
    init_trick: bool = False
    # 𝜀: float = 1e-5

    # # Subsampling Trick
    # subsampling_threshold_schedule: Optional[Iterable[float]] = None
    # subsampling_power: float = 8  # float('inf') for deterministic sampling
    # min_doc_length: int = 2  # for subsampling partisan neutral words

    # Evaluation
    cherries: Optional[Tuple[str, ...]] = (
        'estate_tax', 'death_tax',
        'undocumented_immigrants', 'illegals',

        'health_care_reform', 'obamacare',
        'public_option', 'governmentrun', 'government_takeover',
        'national_health_insurance', 'welfare_state',
        'singlepayer', 'governmentrun_health_care',
        'universal_health_care', 'socialized_medicine',

        'campaign_spending', 'political_speech',
        'independent_expenditures',

        'recovery_and_reinvestment', 'stimulus_bill',
        'military_spending', 'washington_spending',
        'progrowth', 'create_jobs',

        'unborn', 'fetus',
        'prochoice', 'proabortion',
        'family_planning',
    )

    # Housekeeping
    export_error_analysis: Optional[int] = 1  # per epoch
    update_tensorboard: int = 1000  # per batch
    print_stats: int = 10_000  # per batch
    eval_dev_set: int = 50_000  # per batch
    progress_bar_refresh_rate: int = 5  # per second
    reload_path: Optional[str] = None
    clear_tensorboard_log_in_output_dir: bool = True
    delete_all_exisiting_files_in_output_dir: bool = False
    auto_save_per_epoch: Optional[int] = 1
    auto_save_if_interrupted: bool = False
    export_tensorboard_embedding_projector: bool = False

    def __post_init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-i', '--input-dir', action='store', type=str)
        parser.add_argument(
            '-o', '--output-dir', action='store', type=str)
        parser.add_argument(
            '-p', '--pretrained-embedding', action='store', type=str)
        parser.add_argument(
            '-d', '--device', action='store', type=str)

        parser.add_argument(
            '-a', '--architecture', action='store', type=str)
        parser.add_argument(
            '-dd', '--deno-delta', action='store', type=float)
        parser.add_argument(
            '-dg', '--deno-gamma', action='store', type=float)
        parser.add_argument(
            '-cd', '--cono-delta', action='store', type=float)
        parser.add_argument(
            '-cg', '--cono-gamma', action='store', type=float)

        parser.add_argument(
            '-b', '--batch-size', action='store', type=int)
        parser.add_argument(
            '-e', '--num-epochs', action='store', type=int)
        parser.add_argument(
            '--encoded-size', action='store', type=int)
        parser.add_argument(
            '--hidden-size', action='store', type=int)
        parser.parse_args(namespace=self)

        if self.architecture == 'L1':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.ReLU())
        elif self.architecture == 'L2':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 200),
                nn.ReLU(),
                nn.Linear(200, 300),
                nn.ReLU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 200),
                nn.ReLU(),
                nn.Linear(200, 300),
                nn.ReLU())
        elif self.architecture == 'L4':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 200),
                nn.ReLU(),
                nn.Linear(200, 150),
                nn.ReLU(),
                nn.Linear(150, 200),
                nn.ReLU(),
                nn.Linear(200, 300),
                nn.ReLU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 200),
                nn.ReLU(),
                nn.Linear(200, 150),
                nn.ReLU(),
                nn.Linear(150, 200),
                nn.ReLU(),
                nn.Linear(200, 300),
                nn.ReLU())
        else:
            raise ValueError('Unknown architecture argument.')


def main() -> None:
    config = RecomposerConfig()
    black_box = RecomposerExperiment(config)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    main()
