
from dataclasses import dataclass

from .stage import Stage
from ...implements.retool import ReFrame


@dataclass
class Clean(Stage):

    technique : str = ''
    parameters : object = None
    name : str = 'clean'
    auto_prepare : bool = True

    def __post_init__(self):
        self.technique_names = ['cleaners']
        super().__post_init__()
        return self

    def _load_cleaners(self, settings):
        technique = ReFrame(file_path = settings.file_path,
                            compile_keys = settings.compile_keys,
                            out_prefix = settings.out_prefix,
                            source_column = settings.source_column)
        return technique

    def _make_cleaners(self, settings):
        technique = ReFrame(keys = settings.keys,
                            values = settings.values,
                            flags = settings.flags,
                            compile_keys = settings.compile_keys,
                            out_prefix = settings.out_prefix,
                            source_column = settings.source_column)
        return technique

    def start(self, df):
        for technique_name in self.technique_names:
            for section, technique in getattr(self, technique_name).items():
                df = technique.match(df)
        return df