import argparse
import random
import pickle
import os
from dataclasses import dataclass
from typing import Tuple, List, Dict, Counter, Optional

import torch
from torch import nn
from torch.nn.utils import rnn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import editdistance  # for excluding trivial nearest neighbors

# from sklearn.metrics import pairwise
# from sklearn.metrics import confusion_matrix
# from matplotlib import pyplot as plt
# import seaborn as sns

from utils.experiment import Experiment
from utils.improvised_typing import Scalar, Vector, Matrix, R3Tensor
# from utils.word_similarity import all_wordsim
from evaluations import intrinsic_eval

random.seed(42)
torch.manual_seed(42)
# sns.set()


class Decomposer(nn.Module):

    def __init__(
            self,
            config: 'DecomposerConfig',
            data: 'LabeledSentences'):
        super().__init__()
        vocab_size = len(data.word_to_id)
        # self.deno_to_id = data.deno_to_id  # only for getting baseline accuracy
        self.id_to_deno = data.id_to_deno  # only for error analysis
        # self.graph_labels = [
        #     data.id_to_deno[i]
        #     for i in range(len(data.id_to_deno))]

        # Initialize Embedding
        if config.pretrained_embedding is not None:
            config.embed_size = data.pretrained_embedding.shape[1]
            self.embedding = nn.EmbeddingBag.from_pretrained(
                data.pretrained_embedding, mode='sum')
        else:
            self.embedding = nn.EmbeddingBag(vocab_size, config.embed_size)
            init_range = 1.0 / config.embed_size
            nn.init.uniform_(self.embedding.weight.data, -init_range, init_range)
        self.embedding.weight.requires_grad = not config.freeze_embedding

        # Decomposer
        # self.encoder = architecture
        self.beta = config.beta
        self.delta = config.deno_delta
        self.gamma = config.deno_gamma
        self.device = config.device

        num_deno_classes = len(data.deno_to_id)
        num_cono_classes = 2
        repr_size = 300

        # Dennotation Loss: Skip-Gram Negative Sampling
        # self.deno_decoder = nn.Linear(repr_size, num_deno_classes)
        self.deno_decoder = config.deno_architecture
        assert self.deno_decoder[0].in_features == repr_size
        assert self.deno_decoder[-2].out_features == num_deno_classes

        # Connotation Loss: Party Classifier
        # self.cono_decoder = nn.Linear(repr_size, num_cono_classes)
        self.cono_decoder = config.cono_architecture
        assert self.cono_decoder[-2].out_features == num_cono_classes
        self.to(self.device)

        # Initailize neighbor cono homogeneity eval partisan vocabulary
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.counts = data.counts  # saved just in case
        self.grounding = data.grounding
        Dem_ids: List[int] = []
        GOP_ids: List[int] = []
        neutral_ids: List[int] = []
        neutral_bound = 0.25
        GOP_lower_bound = 0.5 + neutral_bound
        Dem_upper_bound = 0.5 - neutral_bound
        for word_id, word in self.id_to_word.items():
            R_ratio = self.grounding[word]['R_ratio']
            if R_ratio > GOP_lower_bound:
                GOP_ids.append(word_id)
            elif R_ratio < Dem_upper_bound:
                Dem_ids.append(word_id)
            else:
                neutral_ids.append(word_id)
        neutral_ids = random.sample(neutral_ids, 2000)  # NOTE
        self.Dem_ids = torch.tensor(Dem_ids)
        self.GOP_ids = torch.tensor(GOP_ids)
        self.neutral_ids = torch.tensor(neutral_ids)
        print(f'{len(GOP_ids)} capitalists\n'
              f'{len(Dem_ids)} socialists\n'
              f'{len(neutral_ids)} neoliberal shills\n')

    def forward(
            self,
            seq_word_ids: Matrix,
            deno_labels: Vector,
            cono_labels: Vector
            ) -> Scalar:
        seq_repr: Matrix = self.embedding(seq_word_ids)  # sum EmbedBag

        deno_logits = self.deno_decoder(seq_repr)
        deno_loss = nn.functional.cross_entropy(deno_logits, deno_labels)

        cono_logits = self.cono_decoder(seq_repr)
        cono_loss = nn.functional.cross_entropy(cono_logits, cono_labels)

        decomposer_loss = (self.delta * deno_loss +
                           self.gamma * cono_loss +
                           self.beta)
        return decomposer_loss, deno_loss, cono_loss

    def predict(self, seq_word_ids: Vector) -> Vector:
        self.eval()
        with torch.no_grad():
            seq_repr: Matrix = self.embedding(seq_word_ids)  # sum EmbedBag

            deno = self.deno_decoder(seq_repr)
            cono = self.cono_decoder(seq_repr)

            deno_conf = nn.functional.softmax(deno, dim=1)
            cono_conf = nn.functional.softmax(cono, dim=1)
        self.train()
        return deno_conf, cono_conf

    def accuracy(
            self,
            seq_word_ids: Matrix,
            deno_labels: Vector,
            cono_labels: Vector,
            error_analysis_path: Optional[str] = None
            ) -> Tuple[float, float]:
        deno_conf, cono_conf = self.predict(seq_word_ids)
        deno_predictions = deno_conf.argmax(dim=1)
        cono_predictions = cono_conf.argmax(dim=1)
        # # Random Guess Baseline
        # deno_predictions = torch.randint_like(deno_labels, high=len(self.deno_to_id))
        # # Majority Class Baseline
        # majority_label = self.deno_to_id['Health']
        # deno_predictions = torch.full_like(deno_labels, majority_label)

        deno_correct_indicies = deno_predictions.eq(deno_labels)
        cono_correct_indicies = cono_predictions.eq(cono_labels)
        deno_accuracy = deno_correct_indicies.float().mean().item()
        cono_accuracy = cono_correct_indicies.float().mean().item()

        if error_analysis_path:
            analysis_file = open(error_analysis_path, 'w')
            analysis_file.write('pred_conf\tpred\tlabel_conf\tlabel\tseq\n')
            output = []
            for pred_confs, pred_id, label_id, seq_ids in zip(
                    deno_conf, deno_predictions, deno_labels, seq_word_ids):
                pred_conf = f'{pred_confs[pred_id].item():.4f}'
                label_conf = f'{pred_confs[label_id].item():.4f}'
                pred = self.id_to_deno[pred_id.item()]
                label = self.id_to_deno[label_id.item()]
                seq = ' '.join([self.id_to_word[i.item()] for i in seq_ids])
                output.append((pred_conf, pred, label_conf, label, seq))
            # output.sort(key=lambda t: t[1], reverse=True)
            for stuff in output:
                analysis_file.write('\t'.join(stuff) + '\n')

        # if error_analysis_path:  # confusion  matrix
        #     cf_mtx = confusion_matrix(deno_labels.cpu(), deno_predictions.cpu())
        #     fig, ax = plt.subplots(figsize=(20, 20))
        #     sns.heatmap(
        #         cf_mtx, annot=True, robust=True, ax=ax, cbar=False, fmt='d', linewidths=.5,
        #         mask=np.equal(cf_mtx, 0),
        #         xticklabels=self.graph_labels, yticklabels=self.graph_labels)
        #     ax.set_xlabel('Predicted Label')
        #     ax.set_ylabel('True Label')
        #     with open(error_analysis_path, 'wb') as file:
        #         fig.savefig(file, dpi=300, bbox_inches='tight')

        return deno_accuracy, cono_accuracy

    def export_embedding(
            self, device: Optional[torch.device] = None
            ) -> Matrix:
        # all_vocab_ids = torch.arange(
        #     self.embedding.num_embeddings, dtype=torch.long)
        # if not device:
        #     device = self.device
        # all_vocab_ids = all_vocab_ids.to(device)

        # self.eval()
        # with torch.no_grad():
        #     embed = self.embedding(all_vocab_ids)
        # self.train()
        return self.embedding.weight.detach()

    def neighbor_heterogeneity_continuous(
            self,
            query_ids: Vector,
            top_k: int = 10
            ) -> float:
        query_ids = query_ids.to(self.device).unsqueeze(1)
        query_embed = self.embedding(query_ids)

        with torch.no_grad():
            top_neighbor_ids = [
                nn.functional.cosine_similarity(
                    q.view(1, -1), self.embedding.weight).argsort(descending=True)
                for q in query_embed]

        homogeneity = []
        for query_index, sorted_target_indices in enumerate(top_neighbor_ids):
            query_id = query_ids[query_index].item()
            query_word = self.id_to_word[query_id]
            num_neighbors = 0

            query_R_ratio = self.grounding[query_word]['R_ratio']
            freq_ratio_distances = []
            for sort_rank, target_id in enumerate(sorted_target_indices):
                target_id = target_id.item()
                if num_neighbors == top_k:
                    break
                if query_id == target_id:
                    continue
                # target_id = target_ids[target_index]  # target is always all embed
                target_words = self.id_to_word[target_id]
                if editdistance.eval(query_word, target_words) < 3:
                    continue
                num_neighbors += 1
                target_R_ratio = self.grounding[target_words]['R_ratio']
                # freq_ratio_distances.append((target_R_ratio - query_R_ratio) ** 2)
                freq_ratio_distances.append(abs(target_R_ratio - query_R_ratio))
            # homogeneity.append(np.sqrt(np.mean(freq_ratio_distances)))
            homogeneity.append(np.mean(freq_ratio_distances))
        return np.mean(homogeneity)

    def neighbor_homogeneity_discrete(
            self,
            query_ids: Vector,
            top_k: int = 10
            ) -> float:
        query_ids = query_ids.to(self.device)
        query_embed = self.embedding(query_ids)
        with torch.no_grad():
            # cosine_sim = nn.functional.cosine_similarity(
            #     query_embed, self.embedding.weight.view(1, -1, 300))
            cosine_sim = pairwise.cosine_similarity(
                query_embed.cpu(), self.embedding.weight.cpu())
        # top_neighbor_ids = torch.argsort(cosine_sim, descending=True)
        top_neighbor_ids = np.argsort(-cosine_sim)
        # top_neighbor_ids = top_neighbor_ids[:, 0:top_k]  # excluding itself?

        # output: Dict[int, List[Tuple[int, float]]] = {}
        homogeneity = []
        for query_index, sorted_target_indices in enumerate(top_neighbor_ids):
            query_id = query_ids[query_index].item()
            query_words = self.id_to_word[query_id]
            num_neighbors = 0
            num_same_cono = 0
            for sort_rank, target_id in enumerate(sorted_target_indices):
                if num_neighbors == top_k:
                    break
                if query_id == target_id:
                    continue
                # target_id = target_ids[target_index]  # target is always all embed
                target_words = self.id_to_word[target_id]
                if editdistance.eval(query_words, target_words) < 3:
                    continue
                num_neighbors += 1
                if target_id in query_ids:
                    num_same_cono += 1
            homogeneity.append(num_same_cono / top_k)
        return np.mean(homogeneity)


class LabeledSentences(Dataset):

    def __init__(self, config: 'DecomposerConfig'):
        corpus_path = os.path.join(config.input_dir, 'train_data.pickle')
        with open(corpus_path, 'rb') as corpus_file:
            preprocessed = pickle.load(corpus_file)
        self.word_to_id = preprocessed['word_to_id']
        self.id_to_word = preprocessed['id_to_word']
        self.deno_to_id = preprocessed['deno_to_id']
        self.id_to_deno = preprocessed['id_to_deno']
        # word -> Counter[deno/cono]
        self.grounding: Dict[str, Counter[str]] = preprocessed['grounding']
        # deno/cono -> Counter[word]
        self.counts: Dict[str, Counter[str]] = preprocessed['counts']

        self.train_seq: List[List[int]] = preprocessed['train_sent_word_ids']
        self.train_deno_labels: List[int] = preprocessed['train_deno_labels']
        self.train_cono_labels: List[int] = preprocessed['train_cono_labels']

        self.dev_seq = rnn.pad_sequence(
            [torch.tensor(seq) for seq in preprocessed['dev_sent_word_ids']],
            batch_first=True)
        self.dev_deno_labels = torch.tensor(preprocessed['dev_deno_labels'])
        self.dev_cono_labels = torch.tensor(preprocessed['dev_cono_labels'])

        del preprocessed
        if config.export_tensorboard_embedding_projector:
            self.vocabulary = [
                self.id_to_word[word_id]
                for word_id in range(len(self.word_to_id))]

        if config.pretrained_embedding is not None:
            if config.pretrained_embedding.endswith('txt'):
                self.pretrained_embedding: Matrix = Experiment.load_embedding(
                    config.pretrained_embedding, self.word_to_id)
            elif config.pretrained_embedding.endswith('pt'):
                self.pretrained_embedding = torch.load(
                    config.pretrained_embedding,
                    map_location=config.device)['model'].embedding.weight
            else:
                raise ValueError('Unknown pretrained embedding format.')

        # if config.debug_subset_corpus:
        #     self.train_word_ids = self.train_word_ids[:config.debug_subset_corpus]
        #     self.train_labels = self.train_labels[:config.debug_subset_corpus]

    def __len__(self) -> int:
        return len(self.train_seq)

    def __getitem__(self, index: int) -> Tuple[List[int], int, int]:
        return (
            self.train_seq[index],
            self.train_deno_labels[index],
            self.train_cono_labels[index])

    @staticmethod
    def collate(
            batch: List[Tuple[List[int], int, int]]
            ) -> Tuple[Matrix, Vector, Vector]:
        # seq_word_ids = torch.cat([torch.tensor(w) for w, _, _ in batch])
        seq_word_ids = [torch.tensor(w) for w, _, _ in batch]
        deno_labels = torch.tensor([d for _, d, _ in batch])
        cono_labels = torch.tensor([c for _, _, c in batch])
        return (
            rnn.pad_sequence(seq_word_ids, batch_first=True),
            deno_labels,
            cono_labels)


class DecomposerExperiment(Experiment):

    def __init__(self, config: 'DecomposerConfig'):
        super().__init__(config)
        self.data = LabeledSentences(config)
        self.dataloader = DataLoader(
            self.data,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.data.collate,
            num_workers=config.num_dataloader_threads,
            pin_memory=config.pin_memory)
        self.model = Decomposer(config, self.data)
        # self.load_cherries()

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)  # param.data)

        self.f_deno_optimizer = config.optimizer(
            self.model.deno_decoder.parameters(),
            lr=config.learning_rate)
        self.f_cono_optimizer = config.optimizer(
            self.model.cono_decoder.parameters(),
            lr=config.learning_rate)

        # self.g_deno_optimizer = config.optimizer(
        #     self.model.decomposer_g.deno_decoder.parameters(),
        #     lr=config.learning_rate)
        # self.g_cono_optimizer = config.optimizer(
        #     self.model.decomposer_g.cono_decoder.parameters(),
        #     lr=config.learning_rate)

        # self.recomposer_optimizer = config.optimizer(
        #     self.model.decomposer_g.cono_decoder.parameters(),
        #     lr=config.learning_rate)
        # self.decomposer_optimizer = config.optimizer((
        #     list(self.model.decomposer_f.encoder.parameters()) +
        #     list(self.model.decomposer_g.encoder.parameters())),
        #     lr=config.learning_rate)

        self.decomposer_optimizer = config.optimizer(
            self.model.embedding.parameters(),  # NOTE not frozen embedding, no encoder
            lr=config.learning_rate)

        self.to_be_saved = {
            'config': self.config,
            'model': self.model}

        self.custom_stats_format = (
            'ℒ = {Recomposer/combined_loss:.3f}\t'
            'Dd = {Denotation Decomposer/deno_loss:.3f}\t'
            'Dc = {Denotation Decomposer/cono_loss:.3f}\t'
            # 'Cd = {Connotation Decomposer/deno:.3f}\t'
            # 'Cc = {Connotation Decomposer/cono:.3f}\t'
            # 'R = {Recomposer/recomp:.3f}\t'
            # 'cono accuracy = {Evaluation/connotation_accuracy:.2%}'
        )

    def _train(
            self,
            seq_word_ids: Vector,
            deno_labels: Vector,
            cono_labels: Vector,
            update_encoder: bool,
            grad_clip: float = 5
            ) -> Tuple[float, float, float]:
        self.model.zero_grad()
        L_master, l_f_deno, l_f_cono = self.model(seq_word_ids, deno_labels, cono_labels)

        l_f_deno.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model.deno_decoder.parameters(), grad_clip)
        self.f_deno_optimizer.step()

        self.model.zero_grad()
        l_f_cono.backward(retain_graph=update_encoder)
        nn.utils.clip_grad_norm_(self.model.cono_decoder.parameters(), grad_clip)
        self.f_cono_optimizer.step()

        if update_encoder:
            self.model.zero_grad()
            L_master.backward()
            nn.utils.clip_grad_norm_(self.model.embedding.parameters(), grad_clip)
            # nn.utils.clip_grad_norm_(self.model.decomposer_g.encoder.parameters(), grad_clip)
            self.decomposer_optimizer.step()

        # self.model.zero_grad()
        # l_g_deno.backward(retain_graph=True)
        # nn.utils.clip_grad_norm_(self.model.decomposer_g.deno_decoder.parameters(), grad_clip)
        # self.g_deno_optimizer.step()

        # self.model.zero_grad()
        # l_g_cono.backward(retain_graph=True)
        # nn.utils.clip_grad_norm_(self.model.decomposer_g.cono_decoder.parameters(), grad_clip)
        # self.g_cono_optimizer.step()

        # self.model.zero_grad()
        # l_h.backward()
        # nn.utils.clip_grad_norm_(self.model.decomposer_g.cono_decoder.parameters(), grad_clip)
        # self.recomposer_optimizer.step()

        return (
            L_master.item(),
            l_f_deno.item(),
            l_f_cono.item(),)
        # l_g_deno.item(),
        # l_g_cono.item(),
        # l_h.item())

    def train(self) -> None:
        config = self.config
        # self.save_everything(
        #     os.path.join(self.config.output_dir, f'untrained.pt'))
        # assert False
        epoch_pbar = tqdm(range(1, config.num_epochs + 1), desc='Epochs')
        for epoch_index in epoch_pbar:
            if not config.print_stats:
                batches = enumerate(self.dataloader)
            else:
                batches = tqdm(
                    enumerate(self.dataloader),
                    total=len(self.dataloader),
                    mininterval=config.progress_bar_refresh_rate,
                    desc='Batches')
            for batch_index, batch in batches:
                seq_word_ids = batch[0].to(self.device)
                deno_labels = batch[1].to(self.device)
                cono_labels = batch[2].to(self.device)

                update_encoder = batch_index % config.encoder_grad_update_freq == 0
                L_master, l_f_deno, l_f_cono = self._train(
                    seq_word_ids, deno_labels, cono_labels, update_encoder)

                if batch_index % config.update_tensorboard == 0:
                    deno_accuracy, cono_accuracy = self.model.accuracy(
                        seq_word_ids, deno_labels, cono_labels)
                    stats = {
                        'Denotation Decomposer/deno_loss': l_f_deno,
                        'Denotation Decomposer/cono_loss': l_f_cono,
                        'Denotation Decomposer/accuracy_train_deno': deno_accuracy,
                        'Denotation Decomposer/accuracy_train_cono': cono_accuracy,
                        # 'Connotation Decomposer/deno': l_g_deno,
                        # 'Connotation Decomposer/cono': l_g_cono,
                        # 'Connotation Decomposer/accuracy_train': g_accuracy,
                        # 'Recomposer/recomp': l_h,
                        'Recomposer/combined_loss': L_master
                    }
                    self.update_tensorboard(stats)
                if config.print_stats and batch_index % config.print_stats == 0:
                    self.print_stats(epoch_index, batch_index, stats)
                if batch_index % config.eval_dev_set == 0:
                    deno_accuracy, cono_accuracy = self.model.accuracy(
                        self.data.dev_seq.to(self.device),
                        self.data.dev_deno_labels.to(self.device),
                        self.data.dev_cono_labels.to(self.device))

                    # D_h = self.model.neighbor_homogeneity_discrete(self.model.Dem_ids)
                    # R_hd = self.model.neighbor_homogeneity_discrete(self.model.GOP_ids)
                    # N_hd = self.model.neighbor_homogeneity_discrete(self.model.neutral_ids)

                    # D_h = self.model.neighbor_heterogeneity_continuous(self.model.Dem_ids)
                    R_hc = self.model.neighbor_heterogeneity_continuous(self.model.GOP_ids)
                    N_hc = self.model.neighbor_heterogeneity_continuous(self.model.neutral_ids)

                #     self.cherry_pick()
                #     V_pretrained, V_deno, V_cono, V_recomp = self.model.export_embeddings()
                #     deno_check = all_wordsim.mean_delta(V_deno, V_pretrained, self.data.id_to_word)
                #     cono_check = all_wordsim.mean_delta(V_cono, V_pretrained, self.data.id_to_word)
                #     recomp_check = all_wordsim.mean_delta(V_recomp, V_pretrained, self.data.id_to_word)
                    self.update_tensorboard({
                        # 'Denotation Decomposer/nonpolitical_word_sim_cf_pretrained': deno_check,
                        'Denotation Decomposer/accuracy_dev_deno': deno_accuracy,
                        # 'Connotation Decomposer/nonpolitical_word_sim_cf_pretrained': cono_check,
                        'Denotation Decomposer/accuracy_dev_cono': cono_accuracy,
                        # 'Intrinsic Evaluation/Dem_neighbor_homogenity': D_h,
                        # 'Intrinsic Evaluation/GOP_neighbor_homogenity': R_hd,
                        'Intrinsic Evaluation/GOP_neighbor_heterogeneity_continous': R_hc,
                        # 'Intrinsic Evaluation/neutral_neighbor_homogenity': N_hd,
                        'Intrinsic Evaluation/neutral_neighbor_heterogeneity_continous': N_hc,
                        # 'Recomposer/nonpolitical_word_sim_cf_pretrained': recomp_check}
                    })
                self.tb_global_step += 1
            # End Batches
            # self.lr_scheduler.step()
            if config.print_stats:
                self.print_timestamp()
            self.auto_save(epoch_index)

            if config.export_error_analysis:
                if (epoch_index % config.export_error_analysis == 0
                        or epoch_index == 1):
                    # self.model.all_vocab_connotation(os.path.join(
                    #     config.output_dir, f'vocab_cono_epoch{epoch_index}.txt'))
                    analysis_path = os.path.join(
                        config.output_dir, f'error_analysis_epoch{epoch_index}.tsv')
                    deno_accuracy, cono_accuracy = self.model.accuracy(
                        self.data.dev_seq.to(self.device),
                        self.data.dev_deno_labels.to(self.device),
                        self.data.dev_cono_labels.to(self.device),
                        error_analysis_path=analysis_path)

            # if config.export_tensorboard_embedding_projector:
            #     V_pretrained, V_deno, V_cono, V_recomp = self.model.export_embeddings()
            #     self.tensorboard.add_embedding(
            #         V_deno, self.data.vocabulary, global_step=epoch_index, tag='deno_embed')
            #     self.tensorboard.add_embedding(
            #         V_cono, self.data.vocabulary, global_step=epoch_index, tag='cono_embed')
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

    def correlate_sim_deltas(self) -> float:
        label_deltas = []
        model_deltas = []

        for pair in self.phrase_pairs:
            try:
                sim = model.cosine_similarity(pair.query, pair.neighbor)
                ref_sim = ref_model.cosine_similarity(pair.query, pair.neighbor)
            except KeyError:
                continue
            model_delta = sim - ref_sim
            model_deltas.append(model_delta)
            label_deltas.append(pair.deno_sim - pair.cono_sim)

            if verbose:
                print(f'{pair.deno_sim}  {pair.cono_sim}  {ref_sim:.2%}  {sim:.2%}  '
                    f'{pair.query}  {pair.neighbor}')

        median = np.median(model_deltas)
        mean = np.mean(model_deltas)
        stddev = np.std(model_deltas)
        rho, _ = spearmanr(model_deltas, label_deltas)
        return rho, median, mean, stddev


@dataclass
class DecomposerConfig():
    # Essential
    input_dir: str = '../data/processed/bill_mentions/topic_deno'
    # input_dir: str = '../data/processed/bill_mentions/title_deno'

    output_dir: str = '../results/debug'
    device: torch.device = torch.device('cuda')
    debug_subset_corpus: Optional[int] = None
    # dev_holdout: int = 5_000
    # test_holdout: int = 10_000
    num_dataloader_threads: int = 12
    pin_memory: bool = True

    beta: float = 5
    # Denotation Decomposer
    decomposed_deno_size: int = 300
    deno_delta: float = 1  # denotation weight 𝛿
    deno_gamma: float = -1  # connotation weight 𝛾

    # Conotation Decomposer
    decomposed_cono_size: int = 300
    cono_delta: float = -0.02  # denotation weight 𝛿
    cono_gamma: float = 1  # connotation weight 𝛾

    # Recomposer
    recomposer_rho: float = 1
    # dropout_p: float = 0.1

    architecture: str = 'L1'
    batch_size: int = 128
    embed_size: int = 300
    num_epochs: int = 50
    encoder_grad_update_freq: int = 4  # per adversary update

    # pretrained_embedding: Optional[str] = None
    pretrained_embedding: Optional[str] = '../data/pretrained_word2vec/for_real.txt'
    # '../results/analysis/L4 0c/epoch50.pt'
    freeze_embedding: bool = False  # NOTE
    # window_radius: int = 5
    # num_negative_samples: int = 10
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    # optimizer: torch.optim.Optimizer = torch.optim.SGD
    learning_rate: float = 1e-4
    # momentum: float = 0.5
    # lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    # num_prediction_classes: int = 5
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
    print_stats: Optional[int] = 10_000  # per batch
    eval_dev_set: int = 10_000  # per batch
    progress_bar_refresh_rate: int = 5  # per second
    suppress_stdout: bool = False  # during hyperparameter tuning
    reload_path: Optional[str] = None
    clear_tensorboard_log_in_output_dir: bool = True
    delete_all_exisiting_files_in_output_dir: bool = False
    auto_save_per_epoch: Optional[int] = 10
    auto_save_if_interrupted: bool = False
    export_tensorboard_embedding_projector: bool = False

    def __post_init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-i', '--input-dir', action='store', type=str)
        parser.add_argument(
            '-o', '--output-dir', action='store', type=str)
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
            '-lr', '--learning-rate', action='store', type=float)
        parser.add_argument(
            '-bs', '--batch-size', action='store', type=int)
        parser.add_argument(
            '-ep', '--num-epochs', action='store', type=int)
        parser.add_argument(
            '-pe', '--pretrained-embedding', action='store', type=str)
        parser.parse_args(namespace=self)

        if self.architecture == 'L1':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 41),
                nn.SELU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 2),
                nn.SELU())
        elif self.architecture == 'L2':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 41),
                nn.SELU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 2),
                nn.SELU())
        elif self.architecture == 'L4':
            self.deno_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 41),
                nn.SELU())
            self.cono_architecture = nn.Sequential(
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 300),
                nn.SELU(),
                nn.Linear(300, 2),
                nn.SELU())
        else:
            raise ValueError('Unknown architecture argument.')


def main() -> None:
    config = DecomposerConfig()
    black_box = DecomposerExperiment(config)
    with black_box as auto_save_wrapped:
        auto_save_wrapped.train()


if __name__ == '__main__':
    main()
