import os
import pprint
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Set, Tuple, List, Dict, Optional, Any, no_type_check

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


@dataclass
class ExperimentConfig():

    input_dir: str
    output_dir: str
    num_epochs: int
    device: torch.device

    reload_path: Optional[str]
    clear_tensorboard_log_in_output_dir: bool
    delete_all_exisiting_files_in_output_dir: bool
    auto_save_per_epoch: Optional[int]
    auto_save_if_interrupted: bool


class Experiment(ABC):

    def __init__(self, config: ExperimentConfig):
        self.tb_global_step = 0
        self.custom_stats_format = None
        self.config = config
        self.device = config.device

    def __enter__(self) -> 'Experiment':
        config = self.config
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
            print(f'Created new directory {config.output_dir} as output_dir.')
        else:
            print(f'output_dir = {config.output_dir}')
            if config.delete_all_exisiting_files_in_output_dir:
                import shutil
                shutil.rmtree(config.output_dir)
                os.makedirs(config.output_dir)
                print('config.delete_all_exisiting_files_in_output_dir = True')
            elif config.clear_tensorboard_log_in_output_dir:
                import shutil
                stuff = os.listdir(config.output_dir)
                for filename in stuff:
                    if filename.startswith('TB '):
                        tb_log_dir = os.path.join(config.output_dir, filename)
                        shutil.rmtree(tb_log_dir)
                        print(f'Deleted {filename} in output_dir')
        timestamp = datetime.now().strftime("%m-%d %H-%M-%S")
        log_dir = os.path.join(config.output_dir, f'TB {timestamp}')
        self.tensorboard = SummaryWriter(log_dir=log_dir)

        # config_dict = asdict(self.config)
        # config_dict['_model'] = str(self.model)
        # self.tensorboard.add_text(
        #     'config', pprint.pformat(config_dict), global_step=0)
        preview_path = os.path.join(config.output_dir, 'config.txt')
        with open(preview_path, 'w') as preview_file:
            # pprint.pprint(config_dict, preview_file)
            if hasattr(self, 'model'):
                preview_file.write('architecture = ' + str(self.model) + '\n')
            for key, val in asdict(self.config).items():
                preview_file.write(f'{key} = {val}\n')
        return self

    @no_type_check
    def __exit__(self, exception_type, exception_value, traceback) -> None:
        if exception_type is not None:
            print(f'Experiment interrupted.')
            if self.config.auto_save_if_interrupted:
                self.save_everything(
                    os.path.join(self.config.output_dir, f'interrupted.pt'))
        else:
            print('\n✅ Training Complete')
        self.tensorboard.close()

    def auto_save(self, epoch_index: int) -> None:
        if self.config.auto_save_per_epoch:
            interim_save = epoch_index % self.config.auto_save_per_epoch == 0
        final_save = epoch_index == self.config.num_epochs
        if interim_save or final_save:
            self.save_everything(
                os.path.join(self.config.output_dir, f'epoch{epoch_index}.pt'))

    @no_type_check
    def save_state_dict(self, save_path: str) -> None:
        """
        PyTorch's recommended method of saving a model,
        but this requires re-instantiate the model object first, along with
        all of its required arguments, which I find to be finicky;
        therefore, save_everything is used by default.
        """
        payload = {
            'config': self.config,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'lr_scheduler': self.lr_scheduler.state_dict()
        }
        torch.save(payload, save_path)
        tqdm.write(f'💾 state_dict saved to {save_path}\n')

    def reload_state_dict(self) -> Any:
        raise NotImplementedError

    @no_type_check
    def save_everything(
            self,
            save_path: str,
            include_data: bool = False
            ) -> None:
        """
        Directly serialize config, model, optimizer, and lr_scheduler,
        similar to PyTorch's non-recommended method of saving
        an entire model (as opposed to only the state_dict).

        Unfortunately, the Experiment class itself is not picklable,
        even with the dill module.
        """
        torch.save(self.to_be_saved, save_path, pickle_protocol=-1)
        tqdm.write(f'💾 Experiment saved to {save_path}')

    @staticmethod
    def reload_everything(
            reload_path: str,
            device: torch.device
            ) -> Tuple[Any, ...]:
        print(f'Reloading model and config from {reload_path}')
        return torch.load(reload_path, map_location=device)

    def print_stats(
            self,
            epoch_index: int,
            batch_index: int,
            stats: Dict
            ) -> None:
        tqdm.write(
            f'Epoch {epoch_index} Batch {batch_index:,}:',
            end='\t')
        if self.custom_stats_format:
            tqdm.write(self.custom_stats_format.format_map(stats))
        else:
            for key, val in stats.items():
                tqdm.write(f'{key} = {val:.3f}', end='\t')
            tqdm.write('')

    def update_tensorboard(
            self,
            stats: Dict[str, Any],
            increment_global_step: bool = False,
            ) -> None:
        """
        Cannot simply use **kwargs because TensorBoard uses slashes to
        organize scope, and slashes are not allowed as Python variable names.
        """
        for key, val in stats.items():
            self.tensorboard.add_scalar(key, val, self.tb_global_step)
        if increment_global_step:
            self.tb_global_step += 1

    def print_timestamp(self, epoch_index: int) -> None:
        # timestamp = datetime.now().strftime('%-I:%M %p')
        timestamp = datetime.now().strftime('%X')
        tqdm.write(
            f'{timestamp}, '
            f'Epoch {epoch_index}, '
            f'TensorBoard Global Step = {self.tb_global_step:,}')

    @abstractmethod
    def train(self) -> Any:
        pass

    # @staticmethod
    # def load_embedding(
    #         in_path: str,
    #         word_to_id: Dict[str, int]
    #         ) -> torch.Tensor:
    #     if in_path.endswith('txt'):
    #         return self.load_txt_embedding(in_path, word_to_id)
    #     elif in_path.endswith('pt'):
    #         raise NotImplementedError('To-Do: check matching word_to_id')
    #         # return torch.load(
    #         #     in_path, map_location=config.device)['model'].embedding.weight
    #     else:
    #         raise ValueError('Unknown pretrained embedding format.')

    @staticmethod
    def uniform_init_embedding(
            vocab_size: int,
            embed_size: int
            ) -> torch.Tensor:
        embedding = nn.Embedding(vocab_size, embed_size)
        init_range = 1.0 / embed_size
        nn.init.uniform_(embedding.weight.data, -init_range, init_range)
        return embedding

    @staticmethod
    def load_txt_embedding(
            in_path: str,
            word_to_id: Dict[str, int],
            verbose: bool = True
            ) -> torch.Tensor:
        """
        Load pretrained emebdding from a plain text file, where the first line
        specifies the vocabulary size and embedding dimensions.

        The word_to_id argument should reflect the tokens in the training corpus,
        the word ids of the pretrained matrix will be converted to match that
        of the corpus.
        """
        real_vocab_size = len(word_to_id)
        print(f'Loading pretrained embedding from {in_path}', flush=True)
        pretrained: Dict[str, List[float]] = {}
        with open(in_path) as file:
            PE_vocab_size, embed_size = file.readline().split()
            embed_size = int(embed_size)
            print(f'Pretrained vocab size = {PE_vocab_size}, embed dim = {embed_size}')
            for line in file:
                line = line.split()  # type: ignore
                word = line[0]
                pretrained[word] = list(map(float, line[-embed_size:]))
        # if len(pretrained) < real_vocab_size:  # len(pretrained) != PE_vocab_size
        #     print(f'Note: pretrained vocab size = {len(pretrained):,}'
        #           f' < {real_vocab_size:,} = training corpus vocab size')

        out_of_vocabulary: Set[str] = set()
        embedding = torch.zeros(real_vocab_size, embed_size, dtype=torch.float32)
        init_range = 1.0 / embed_size  # initialize matrix for OOV rows
        nn.init.uniform_(embedding, -init_range, init_range)
        for word, word_id in word_to_id.items():
            try:
                vector = pretrained[word]
                embedding[word_id] = torch.tensor(vector)
            except KeyError:
                out_of_vocabulary.add(word)
        if out_of_vocabulary:
            print('The following words in the training corpus are out of '
                  'the vocabulary of the given pretrained embedding: ')
            print(out_of_vocabulary)
        return nn.Embedding.from_pretrained(torch.tensor(embedding))
        # Unsafe
        # if word_to_id is None:
        #     embedding: List[List[float]] = []
        #     id_generator = 0
        #     word_to_id: Dict[str, int] = {}  # type: ignore
        #     id_to_word: Dict[int, str] = {}
        #     for line in file:
        #         line = line.split()
        #         word = line[0]
        #         embedding.append(list(map(float, line[-embed_size:])))
        #         word_to_id[word] = id_generator
        #         id_to_word[id_generator] = word
        #         id_generator += 1
        #     return torch.tensor(embedding), word_to_id, id_to_word

    # @staticmethod
    # def convert_word_ids(
    #         corpus: List[int],
    #         corpus_id_to_word: Dict[int, str],
    #         pretrained_word_to_id: Dict[str, int],
    #         OOV_token: Optional[str] = None
    #         ) -> List[int]:
    #     """
    #     Convert word_ids constructed from preprocessed corpus
    #     to the word_ids used by pretrained_embedding.
    #     """
    #     converted = []
    #     out_of_vocabulary = set()
    #     for corpus_word_id in tqdm(corpus, desc='Converting word_ids'):
    #         word = corpus_id_to_word[corpus_word_id]
    #         try:
    #             converted.append(pretrained_word_to_id[word])
    #         except KeyError:
    #             out_of_vocabulary.add(word)
    #             if OOV_token:
    #                 converted.append(pretrained_word_to_id[OOV_token])

    #     if out_of_vocabulary:
    #         print('The following words in the corpus are out of'
    #               'the vocabulary of the given pretrained embedding.')
    #         print(out_of_vocabulary)
    #         if not OOV_token:
    #             raise KeyError
    #     return converted
