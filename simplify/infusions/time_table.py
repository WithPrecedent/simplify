
from dataclasses import dataclass

import pandas as pd


@dataclass
class TimeTable(object):
    """Stores external data where a unit of time is one dimension.
    """
    settings : object = None
    filer : object = None
    df : object = None

    def __post_init__(self):
        return self