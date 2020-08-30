from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

@dataclass
class Sentence():
    tokens: List[str] = field(default_factory=list)
    normalized_tokens: List[str] = field(default_factory=list)
    underscored_tokens: List[str] = field(default_factory=list)
    subsampled_tokens: List[str] = field(default_factory=list)
    numerical_tokens: List[int] = field(default_factory=list)


@dataclass
class LabeledDoc():
    uid: str
    title: str
    url: str
    party: str  # left, left-center, least, right-center, right
    # partisan: bool
    referent: str
    text: str
    date: Optional[str] = None
    sentences: Optional[List[Sentence]] = None


@dataclass
class GroundedWord():
    word: str
    id: int
    deno: Optional[str]
    cono: str
    # cono_ratio: np.ndarray
    # R_ratio
    # cono_PMI: np.ndarray
    # TODO majority deno

#     def __post_init__(self) -> None:
#         self.word_id: int = WTI[self.word]
#         metadata = sub_PE_GD[self.word]
#         self.freq: int = metadata['freq']
#         self.R_ratio: float = metadata['R_ratio']
#         self.majority_deno: int = metadata['majority_deno']

#         self.PE_neighbors = self.neighbors(PE)

#     def deno_ground(self, embed, top_k=10):
#         self.neighbors: List[str] = nearest_neighbors()

    # def __str__(self) -> str:
    #     return (
    #         f'{self.word}\t'
    #         f'{self.cono_freq}\t'
    #         f'{np.around(self.cono_ratio, 4)}\t'
    #         f'{np.around(self.cono_PMI, 4)}')

    def init_extra(self) -> None:
        self.freq = np.sum(self.cono_freq)
        self.R_ratio = self.cono_freq[2] / (self.cono_freq[0] + self.cono_freq[2])
