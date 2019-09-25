
from dataclasses import dataclass

from simplify.core.base import SimpleTechnique


@dataclass
class TorchModel(SimpleTechnique):
    """Applies machine learning algorithms based upon user selections."""
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'torch'

    def __post_init__(self):
        super().__post_init__()
        return self
