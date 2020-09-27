from pathlib import Path
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

    input_dir: Path
    output_dir: Path
    num_epochs: int
    device: torch.device

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
        try:
            from rich.traceback import install
            install()
        except ImportError:
            print('Note: Rich text traceback avaliable by pip install rich')

    def __enter__(self) -> 'Experiment':
        config = self.config
        if not isinstance(config.output_dir, Path):
            raise TypeError('config.output_dir must be a pathlib.Path')
        if not Path.exists(config.output_dir):
            Path.mkdir(config.output_dir, parents=True)
            print(f'Created new directory {config.output_dir} as output_dir.')
        else:
            print(f'output_dir = {config.output_dir}')
            if config.delete_all_exisiting_files_in_output_dir:
                import shutil
                shutil.rmtree(config.output_dir)
                Path.mkdir(config.output_dir, parents=True)
                print('config.delete_all_exisiting_files_in_output_dir = True')
            elif config.clear_tensorboard_log_in_output_dir:
                import shutil
                for file in config.output_dir.iterdir():
                    if file.name.startswith('TB '):
                        tb_log_dir = config.output_dir / file.name
                        shutil.rmtree(tb_log_dir)
                        print(f'Deleted {file.name} in output_dir')

        timestamp = datetime.now().strftime("%m-%d %H-%M-%S")
        log_dir = config.output_dir / f'TB {timestamp}'
        self.tensorboard = SummaryWriter(log_dir=log_dir)

        config_dict = asdict(self.config)
        config_dict['pytorch_version'] = torch.__version__
        config_dict['cuda_version'] = torch.version.cuda
        config_dict['timestamp'] = timestamp
        if hasattr(self, 'model'):
            config_dict['_model'] = str(self.model)
        preview_path = config.output_dir / 'config.txt'
        with open(preview_path, 'w') as preview_file:
            # pprint.pprint(config_dict, preview_file)
            for key, val in config_dict.items():
                preview_file.write(f'{key} = {val}\n')
                # if not isinstance(val, (int, float, str, bool, torch.Tensor)):
                #     config_dict[key] = str(val)  # for TensorBoard HParams
        # self.tensorboard.add_hparams(hparam_dict=config_dict, metric_dict={})
        return self

    @no_type_check
    def __exit__(self, exception_type, exception_value, traceback) -> None:
        if exception_type is not None:
            print(f'Experiment interrupted.')
            if self.config.auto_save_if_interrupted:
                self.save_everything(self.config.output_dir / f'interrupted.pt')
        else:
            print('\n✅ Training Complete')
        self.tensorboard.close()

    def auto_save(self, epoch_index: int) -> None:
        if self.config.auto_save_per_epoch:
            interim_save = epoch_index % self.config.auto_save_per_epoch == 0
        else:
            interim_save = False
        initial_save = epoch_index == 1
        final_save = epoch_index == self.config.num_epochs
        if initial_save or interim_save or final_save:
            self.save_everything(
                self.config.output_dir / f'epoch{epoch_index}.pt')

    @no_type_check
    def save_state_dict(self, save_path: str) -> None:
        """
        PyTorch's recommended method of saving a model,
        but this requires re-instantiate the model object first,
        along with all of its required arguments, which I find to be finicky;
        thus save_everything is used by default.
        """
        cucumbers = {
            'model': self.model.state_dict(),
            'config': self.config,
            # TODO needs data to init model too
            # 'optimizer': self.optimizer.state_dict(),
            # 'lr_scheduler': self.lr_scheduler.state_dict()
        }
        torch.save(cucumbers, save_path, pickle_protocol=-1)
        tqdm.write(f'💾 state_dict saved to {save_path}\n')

    def reload_state_dict(self) -> Any:
        raise NotImplementedError

    @no_type_check
    def save_everything(
            self,
            save_path: str,
            # include_data: bool = False
            ) -> None:
        """
        PyTorch's non-recommended method of saving an entire model
        (as opposed to only the state_dict).

        Unfortunately, the Experiment class itself is not picklable,
        even with the dill module.
        """
        torch.save(self.model, save_path, pickle_protocol=-1)
        tqdm.write(f'💾 Experiment saved to {save_path}')

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
            manual_step: Optional[int] = None
            ) -> None:
        """
        Cannot simply use **kwargs because TensorBoard uses slashes to
        organize scope, and slashes are not allowed as Python variable names.
        """
        if manual_step:
            for key, val in stats.items():
                self.tensorboard.add_scalar(key, val, manual_step)
        else:
            for key, val in stats.items():
                self.tensorboard.add_scalar(key, val, self.tb_global_step)

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
            special_tokens: Optional[List[str]] = None,
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
            print(f'Pretrained vocab size = {int(PE_vocab_size):,}, '
                  f'embed dim = {embed_size}')
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
            print('\nThe following words in the training corpus are out of '
                  'the vocabulary of the given pretrained embedding: ')
            print(out_of_vocabulary, end='\n\n')
        return nn.Embedding.from_pretrained(embedding)
