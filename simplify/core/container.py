
from abc import ABC, abstractmethod
from dataclasses import dataclass

from simplify.core.base import SimpleClass


@dataclass
class SimpleContainer(SimpleClass, ABC):
    
    def __post_init__(self):
        super().__post_init__()
        # Outputs class status to console if verbose option is selected.
        if self.verbose:
            print('Creating', self)        
        return self

