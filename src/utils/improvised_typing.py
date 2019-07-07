import torch
from typing import Tuple


def _check_rank(declared_rank: int, actual_shape: Tuple[int, ...]) -> bool:
    if declared_rank == len(actual_shape):
        return True
    else:
        raise RuntimeError(
            f'Tensor is declared to be Rank-{declared_rank}, '
            f'but it actually has shape {actual_shape}')
        return False


class Scalar(torch.Tensor):
    def check_rank(self) -> bool:
        return True if _check_rank(0, self.shape) else False


class Vector(torch.Tensor):
    def check_rank(self) -> bool:
        return True if _check_rank(1, self.shape) else False


class Matrix(torch.Tensor):
    def check_rank(self) -> bool:
        return True if _check_rank(2, self.shape) else False


class R3Tensor(torch.Tensor):
    def check_rank(self) -> bool:
        return True if _check_rank(3, self.shape) else False


class R4Tensor(torch.Tensor):
    def check_rank(self) -> bool:
        return True if _check_rank(4, self.shape) else False


class R5Tensor(torch.Tensor):
    def check_rank(self) -> bool:
        return True if _check_rank(5, self.shape) else False
