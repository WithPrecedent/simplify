
from dataclasses import dataclass

from ..model import Algorithm


@dataclass
class TorchModel(Algorithm):
    """Applies machine learning algorithms based upon user selections."""
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'torch'

    def __post_init__(self):
        super().__post_init__()
        return self
