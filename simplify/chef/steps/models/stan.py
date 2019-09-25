
from dataclasses import dataclass

from simplify.core.base import SimpleTechnique

#    from pystan import StanModel


@dataclass
class StanModel(SimpleTechnique):
    """Applies machine learning algorithms based upon user selections."""
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'model'

    def __post_init__(self):
        super().__post_init__()
        return self
