from dataclasses import dataclass
import typing as T


class Polarities:
    POSITIVE = "POS"
    NEGATIVE = "NEG"


@dataclass
class AspectData:
    aspect: str
    opinion: str
    polarity: str


@dataclass
class SampleData:
    text: str
    aspects: T.Optional[T.List[AspectData]] = None
