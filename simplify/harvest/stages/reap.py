
from dataclasses import dataclass

from .stage import Stage
from ...implements.retool import ReOrganize, ReSearch


@dataclass
class Reap(Stage):

    technique : str = ''
    parameters : object = None
    name : str = 'reap'
    auto_prepare : bool = True

    def __post_init__(self):
        self.techniques = {'separate' : ReOrganize,
                           'forage' : ReSearch}
        super().__post_init__()
        return self

    def _create_techniques(self):
        for technique in self.technique_names:
            self._check_techniques(technique)
            technique_instructions = getattr(self.instructions, technique)
            if 'file_path' in technique_instructions:
                method = '_load_' + technique
            else:
                method = '_make_' + technique
            if technique in 'reapers':
                self.techniques.update(
                        {'reapers' : self.instructions.reaper_parameters})
            else:
                for section, settings in technique_instructions.items():
                    self.techniques.update({section : method(settings)})
        return self

    def _load_reapers(self, settings):
        technique = ReOrganize(file_path = settings.file_path,
                             compile_keys = settings.compile_keys,
                             out_prefix = settings.out_prefix)
        return technique

    def _load_threshers(self, settings):
        technique = ReSearch(file_path = settings.file_path,
                           compile_keys = settings.compile_keys,
                           out_prefix = settings.out_prefix)
        return technique

    def _make_reapers(self, settings):
        technique = ReOrganize(keys = settings.keys,
                             values = settings.values,
                             flags = settings.flags,
                             compile_keys = settings.compile_keys,
                             out_prefix = settings.out_prefix)
        return technique

    def _make_threshers(self, settings):
        technique = ReSearch(keys = settings.keys,
                           values = settings.values,
                           flags = settings.flags,
                           compile_keys = settings.compile_keys,
                           out_prefix = settings.out_prefix)
        return technique



    def start(self, df, source = None):
        kwargs = self.parameters
        if self.technique in ['separate']:
            df, source = self.techniques[self.technique](
                    df = df, source = source, **kwargs)
        else:
            df = self.techniques[self.technique](df = df, **kwargs)
        return df, source