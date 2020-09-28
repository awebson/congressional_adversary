from dataclasses import dataclass, field
from typing import List, Counter, Optional

# import numpy as np

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
    majority_deno: Optional[str] = None
    majority_cono: Optional[str] = None

    def __str__(self) -> str:
        if self.deno is not None:
            return (
                f'{self.text}\t'
                f'{self.deno}\t'
                f'{self.cono}\t')
        else:
            return (
                f'{self.text}\t'
                f'{self.cono}\t')
