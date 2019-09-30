
from dataclasses import dataclass

from simplify.core.base import SimpleTechnique


@dataclass
class TorchModel(SimpleTechnique):
    """Applies Torch to data.

    Args:
        technique (str): name of technique.
        parameters (dict): dictionary of parameters to pass to selected
            algorithm.
        name (str): name of class for matching settings in the Idea instance
            and for labeling the columns in files exported by Critic.
        auto_finalize (bool): whether 'finalize' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique : object = None
    parameters : object = None
    auto_finalize : bool = True
    name : str = 'torch'

    def __post_init__(self):
        super().__post_init__()
        return self

    def draft(self):
        super().draft()
        return self
