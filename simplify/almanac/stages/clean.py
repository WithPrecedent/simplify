
from dataclasses import dataclass
import os
import re

from ..blackacre import Blackacre
from ...implements import ReMatch


@dataclass
class Clean(Blackacre):

    origins : object = None
    techniques : object = None
    regexes : object = None
    prefix : str = 'section_'
    file_path : str = ''

    def __post_init__(self):

        return self
