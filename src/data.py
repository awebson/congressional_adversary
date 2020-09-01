from dataclasses import dataclass, field
from typing import List, Counter, Optional

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
    text: str
    # id: int
    deno: Optional[Counter[str]]
    cono: Optional[Counter[str]]
    # majority_deno: Optional[str] = None
    # majority_cono: Optional[str] = None

    # def __post_init__(self) -> None:
    #     if self.deno is not None:
    #         self.majority_deno: str = self.cono.most_common(1)[0][0]
    #     if self.cono is not None:
    #         self.majority_cono: str = self.deno.most_common(1)[0][0]

    def __str__(self) -> str:
        if self.deno is not None:
            return (
                f'{self.text}\t'
                f'{self.deno}\t'
                f'{self.cono}\t'
                # f'{self.majority_cono}\t'
                # f'{np.around(self.cono_ratio, 4)}\t'
                # f'{np.around(self.cono_PMI, 4)}'
            )
        else:
            return (
                f'{self.text}\t'
                f'{self.cono}\t')

    # def init_extra(self) -> None:
    #     self.freq = np.sum(self.cono_freq)
    #     self.R_ratio = self.cono_freq[2] / (self.cono_freq[0] + self.cono_freq[2])
