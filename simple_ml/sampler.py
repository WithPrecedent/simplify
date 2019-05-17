"""
Sampler is a class containing resampling algorithms used in the siMpLify
package.
"""

from dataclasses import dataclass

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE

from step import Step

@dataclass
class Sampler(Step):

    name : str = ''
    params : object = None

    def __post_init__(self):
        super().__post_init__()
        self.options = {'adasyn' : ADASYN,
                        'smote' : SMOTE,
                        'smoteenn' :  SMOTEENN,
                        'smotetomek' : SMOTETomek}
        self.defaults = {}
        self.runtime_params = {'random_state' : self.seed}
        self.initialize()
        return self
