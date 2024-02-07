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

    aspect_ids: T.Optional[T.List[int]] = None
    opinion_ids: T.Optional[T.List[int]] = None

    def __hash__(self) -> int:
        return hash(self.aspect + self.opinion + self.polarity)


@dataclass
class SampleData:
    sample_id: int
    text: str
    aspects: T.Optional[T.List[AspectData]] = None
