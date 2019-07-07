import os
import pickle
import functools
import pprint
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from torch.utils import tensorboard
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


@dataclass
class ExperimentConfig():

    # Essential
    corpus_dir: str
    output_dir: str
    device: torch.device

    # Hyperparameters
    batch_size: int
    num_epochs: int

    optimizer: functools.partial
    lr_scheduler: functools.partial

    reload_state_dict_path: Optional[str]
    reload_experiment_path: Optional[str]
    clear_tensorboard_log_in_output_dir: bool
    delete_all_exisiting_files_in_output_dir: bool
    auto_save_every_epoch: bool

    # Housekeeping
    auto_save_before_quit: bool
    save_to_tensorboard_embedding_projector: bool
    update_tensorboard_per_batch: int
    print_stats_per_batch: int


class Experiment(ABC):

    def __init__(
            self,
            config: ExperimentConfig,
            data: Dataset,
            dataloader: DataLoader,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler._LRScheduler):
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
                    if filename.startswith('tensorboard'):
                        tb_log_dir = os.path.join(config.output_dir, filename)
                        shutil.rmtree(tb_log_dir)
                        print(f'Deleted {filename} in output_dir')

        # if config.reload_model_path:  # TODO also reload optimizer
        #     print(f'Reloading model at {config.reload_model_path}')
        #     self.model.load_state_dict(torch.load(config.reload_model_path))

        # TensorBoard
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        log_dir = os.path.join(config.output_dir, f'tensorboard_{timestamp}')
        self.tensorboard = tensorboard.SummaryWriter(log_dir=log_dir)
        self.config = config
        self.data = data
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def __enter__(self):
        print(f'device = {self.config.device}')
        print(f'model = {self.model}')
        self.config.num_train_examples = len(self.data)
        config = asdict(self.config)
        pprint.pprint(config)
        self.tensorboard.add_text('config', pprint.pformat(config), global_step=0)
        # for key, val in vars(config).items():
        #     self.tensorboard.add_text(str(key), str(val), global_step=0)
        path = os.path.join(self.config.output_dir, 'config.txt')
        with open(path, 'w') as file:
            pprint.pprint(config, file)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if (exception_type is not None
                and self.config.auto_save_before_quit):
            print(f'Experiment interrupted.')
            save_path = os.path.join(
                self.config.output_dir, f'interrupted.pt')
            torch.save(self.model.state_dict(), save_path)
            print(f'💾 model.state_dict saved to {save_path}\n')
        self.tensorboard.close()

    def save_state_dict(
            self,
            epoch_index: int,
            tb_global_step: int,
            ) -> None:
        """PyTorch's recommended method of saving a model"""
        save_path = os.path.join(
            self.config.output_dir, f'epoch{epoch_index}.pt')
        torch.save(self.model.state_dict(), save_path)
        timestamp = datetime.now().strftime("%b %d %H:%M")
        tqdm.write(f'{timestamp}, Epoch {epoch_index}, '
                   f'TensorBoard Global Step {tb_global_step:,}')
        tqdm.write(f'💾 model.state_dict saved to {save_path}\n')

    def save_experiment(
            self,
            epoch_index: int,
            tb_global_step: int,
            ) -> None:
        """
        Save the entire Experiment class, similar to PyTorch's non-recommended
        method of saving an entire model (as opposed to only the state_dict).

        This also includes the optimizer and learning rate scheduler.
        """
        save_path = os.path.join(
            self.config.output_dir, f'epoch{epoch_index}.pt')
        with open(save_path, 'wb') as save_file:
            pickle.dump(self, save_file)
        timestamp = datetime.now().strftime("%b %d %H:%M")
        tqdm.write(f'{timestamp}, Epoch {epoch_index}, '
                   f'TensorBoard Global Step {tb_global_step:,}')
        tqdm.write(f'💾 Experiment saved to {save_path}\n')

    @staticmethod
    def print_stats(loss: float, epoch_index: int, batch_index: int) -> None:
        tqdm.write(f'Epoch {epoch_index}, Batch {batch_index:,}:\t'
                   f'Loss = {loss:.5f}\t')

    @abstractmethod
    def train(self):
        pass

    # @property
    # @abstractmethod
    # def dataloader(self):
    #     pass

    # @property
    # @abstractmethod
    # def model(self) -> torch.nn.Module:
    #     pass

    # @property
    # @abstractmethod
    # def optimizer(self):
    #     pass

    # @property
    # @abstractmethod
    # def lr_scheduler(self):
    #     pass
