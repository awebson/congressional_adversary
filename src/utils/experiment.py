import os
import pprint
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Optional, Any, no_type_check

import torch
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
    auto_save: bool
    auto_save_per_epoch: Optional[int]


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
                    if filename.startswith('tensorboard'):
                        tb_log_dir = os.path.join(config.output_dir, filename)
                        shutil.rmtree(tb_log_dir)
                        print(f'Deleted {filename} in output_dir')

        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        log_dir = os.path.join(config.output_dir, f'tensorboard_{timestamp}')
        self.tensorboard = SummaryWriter(log_dir=log_dir)

        config_dict = asdict(self.config)
        self.tensorboard.add_text(
            'config', pprint.pformat(config_dict), global_step=0)
        preview_path = os.path.join(config.output_dir, 'config.txt')
        with open(preview_path, 'w') as preview_file:
            pprint.pprint(config_dict, preview_file)
        return self

    @no_type_check
    def __exit__(self, exception_type, exception_value, traceback) -> None:
        if exception_type is not None:
            print(f'Experiment interrupted.')
            if self.config.auto_save:
                self.save_everything(
                    os.path.join(self.config.output_dir, f'interrupted.pt'))
        else:
            print('\n✅ Training Complete')
        self.tensorboard.close()

    def auto_save(self, epoch_index: int) -> None:
        if not self.config.auto_save:
            return
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

    def update_tensorboard(self, stats: Dict[str, Any]) -> None:
        """
        Cannot simply use **kwargs because TensorBoard uses slashes to
        organize scope, and slashes are not allowed as Python variable names.
        """
        for key, val in stats.items():
            self.tensorboard.add_scalar(key, val, self.tb_global_step)
        self.tb_global_step += 1

    def print_timestamp(self) -> None:
        # timestamp = datetime.now().strftime('%-I:%M %p')
        timestamp = datetime.now().strftime('%X')
        tqdm.write(
            f'TensorBoard Global Step = {self.tb_global_step:,} at '
            f'{timestamp}\n\n')

    @abstractmethod
    def train(self) -> Any:
        pass
