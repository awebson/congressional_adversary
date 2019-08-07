import os
import pickle
import functools
import pprint
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Optional, Any

import torch
from torch.utils import tensorboard
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


@dataclass
class ExperimentConfig():

    # Essential
    input_dir: str
    output_dir: str
    device: torch.device

    # # Hyperparameters
    # batch_size: int
    # num_epochs: int

    optimizer: functools.partial
    lr_scheduler: functools.partial

    reload_path: Optional[str]
    clear_tensorboard_log_in_output_dir: bool
    delete_all_exisiting_files_in_output_dir: bool
    # auto_save_every_epoch: bool

    # Housekeeping
    auto_save_before_quit: bool
    # tensorboard_embedding_projector: bool
    # update_tensorboard: int  # per batch
    # print_stats: int  # per batch


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

        # Boilerplate
        self.config = config
        self.data = data
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.custom_stats_format = None
        self.tb_global_step = 0

    def __enter__(self) -> 'Experiment':
        print(f'device = {self.config.device}')
        print(f'model = {self.model}')
        # self.config.num_train_examples = len(self.data)  # ?
        config = asdict(self.config)
        self.tensorboard.add_text(
            'config', pprint.pformat(config, indent=4), global_step=0)
        # for key, val in vars(config).items():
        #     self.tensorboard.add_text(str(key), str(val), global_step=0)
        path = os.path.join(self.config.output_dir, 'config.txt')
        with open(path, 'w') as file:
            pprint.pprint(config, file)
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:  # type: ignore
        if (exception_type is not None
                and self.config.auto_save_before_quit):
            print(f'Experiment interrupted.')
            self.save_everything(os.path.join(
                self.config.output_dir, f'interrupted.pt'))
            # torch.save(self.model.state_dict(), save_path)
            # print(f'💾 model.state_dict saved to {save_path}\n')
        self.tensorboard.close()

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
            'lr_scheduler': self.lr_scheduler.state_dict()
        }
        torch.save(payload, save_path)
        tqdm.write(f'💾 state_dict saved to {save_path}\n')

    def reload_state_dict():
        raise NotImplementedError

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
        if include_data:
            payload = {
                'config': self.config,
                'model': self.model,
                'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler,
                'data': self.data,
                'dataloader': self.dataloader
            }
        else:
            payload = {
                'config': self.config,
                'model': self.model,
                'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler
            }
        torch.save(payload, save_path, pickle_protocol=-1)
        tqdm.write(f'💾 Experiment saved to {save_path}\n')

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

    def update_tensorboard(self, stats: Dict) -> None:
        for key, val in stats.items():
            self.tensorboard.add_scalar(key, val, self.tb_global_step)
        self.tb_global_step += 1

    # TODO format duration
    def print_timestamp(self) -> None:
        timestamp = datetime.now().strftime("%b %d %H:%M")
        tqdm.write(
            f'{timestamp}, '
            f'TensorBoard Global Step {self.tb_global_step:,}\n')

    @abstractmethod
    def train(self) -> Any:
        pass
