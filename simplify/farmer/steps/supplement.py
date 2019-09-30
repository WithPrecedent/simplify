
from dataclasses import dataclass

from simplify.core.base import SimpleStep


@dataclass
class Supplement(SimpleStep):
    """Adds new data to similarly structured DataFrame.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_finalize(bool): whether 'finalize' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique : object = None
    parameters : object = None
    name : str = 'supplementer'
    auto_finalize : bool = True

    def __post_init__(self):
        return self

    def draft(self):
        self.options = {}
        return self

    def produce(self, ingredients, sources):
        return ingredients