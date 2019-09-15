
from dataclasses import dataclass

from simplify.core.step import Step


@dataclass
class Animate(Step):
    
    technique : str = ''
    parameters : object = None
    auto_finalize : bool = True
    auto_produce : bool = False
    name : str = 'animator'
    
    def __post_init__(self):
        super().__post_init__()
        return self