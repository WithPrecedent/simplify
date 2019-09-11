
from dataclasses import dataclass

from ..model import Algorithm

#    from pystan import StanModel


@dataclass
class StanModel(Algorithm):
    """Applies machine learning algorithms based upon user selections."""
    technique : str = ''
    parameters : object = None
    auto_prepare : bool = True
    name : str = 'model'

    def __post_init__(self):
        super().__post_init__()
        return self
