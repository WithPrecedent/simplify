
from dataclasses import dataclass

from .harvest_step import HarvestStep
from ...managers import Technique


@dataclass
class Bundle(HarvestStep):
    """Class for combining different datasets."""
    options : object = None
    almanac : object = None
    auto_prepare : bool = True
    name : str = 'bundler'

    def __post_init__(self):
        self.default_options = {'mergers' : Merger}
        super().__post_init__()
        return self

    def _prepare_mergers(self):
        return self

@dataclass
class Merger(Technique):

    index_columns : object = None
    merge_type : str = ''

    def __post_init__(self):
        return self

    def start(self, ingredients, sources):
        return ingredients