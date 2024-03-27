from dataclasses import dataclass
import pathlib
import typing as T

import toml


@dataclass
class BaseRecipe:
    @classmethod
    def from_dict(cls, values: T.Dict) -> "BaseRecipe":
        raise NotImplementedError()
    
    @classmethod
    def from_file(cls, path: pathlib.Path) -> "BaseRecipe":
        return cls.from_dict(toml.load(path))
