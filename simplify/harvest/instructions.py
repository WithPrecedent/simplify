
from dataclasses import dataclass
import os

from .blackacre import Blackacre
from .stages import Sow, Harvest, Clean, Bundle, Deliver

@dataclass
class Instructions(Blackacre):
    """Defines rules for harvesting (reaping and threshing), cleaning,
    bundling, and delivering data as part of the siMpLify Almanac subpackage.

    Attributes:
        menu: an instance of Menu
        prefixes: a dictionary containing section prefixes and datatypes.
        sowers: a dictionary containing section prefixes and sower settings.
        harvesters: a dictionary containing section prefixes and harvester
            settings.
        cleaners: a dictionary containing section prefixes and cleaner
            settings.
        bundlers: a dictionary containing section prefixes and bundler
            settings.
        deliveries: a dictionary containing section prefixes and deilvery
            settings.
    """
    menu : object
    inventory : object
    prefixes : object = None
    sowers : object = None
    harvesters : object = None
    cleaners : object = None
    bundlers : object = None
    deliveries : object = None
    auto_prepare : bool = True

    def __post_init__(self):
        self._set_defaults()
        super().__post_init__()
        return self

    def _conform_datatypes_to_stage(self):
        """Adjusts some of the siMpLify-specific datatypes to the appropriate
        datatype based upon the stage of the Almanac.
        """
        for prefix, datatype in self.prefixes.items():
            if self.stage in ['harvest', 'clean']:
                if datatype in ['category', 'encoder', 'interactor']:
                    self.prefixes[prefix] = str
            elif self.stage in ['deliver', 'cookbook']:
                if datatype in ['list', 'pattern']:
                    self.prefixes[prefix] = 'category'
        return self

    def _set_cleaners(self):
        for key, value in self.cleaners.items():
            file_name = key +  '.csv'
            file_path = os.path.join(self.inventory.cleaners,
                                     self.jurisdiction,
                                     file_name)
            out_prefix = key + '_'
            source = 'section_' + key
            self.cleaners[key].update({'file_path' : file_path,
                                       'out_prefix' : out_prefix,
                                       'source_column' : source})
        return self

    def _set_defaults(self):
        self.setting_types = []
        self.stage_classes = [Sow, Harvest, Clean, Bundle, Deliver]
        for stage_class in self.stage_classes:
            self.setting_types.extend(stage_class.technique_names)
        return self

    def _set_mappers(self):
        return self

    def _set_mergers(self):
        return self

    def _set_reapers(self):
        if not self.reapers:
            file_name = self.source +  '.csv'
            file_path = os.path.join(self.inventory.reapers, file_name)
            self.reapers = {'reapers' : {'file_path' : file_path,
                                          'compile_keys' : True,
                                          'out_prefix' : 'section_'}}
        return self

    def _set_shapers(self):
        return self

    def _set_sowers(self):
        return self

    def _set_streamliners(self):
        return self

    def _set_threshers(self):
        for key, value in self.threshers.items():
            file_name = key +  '.csv'
            file_path = os.path.join(self.inventory.threshers,
                                     self.jurisdiction,
                                     file_name)
            out_prefix = key + '_'
            self.threshers[key].update({'file_path' : file_path,
                                        'compile_keys' : True,
                                        'out_prefix' : out_prefix})
        return self

    def implement(self, stage, data_source, jurisdiction, case_type):
        self.stage = stage
        self.data_source = data_source
        self.jurisdiction = jurisdiction
        self.case_type = case_type
        self.index_column = 'index_' + self.data_source
        self._conform_datatypes_to_stage()
        for setting_type in self.setting_types:
            if hasattr(self, '_set' + setting_type):
                getattr(self, '_set' + setting_type)()
        return self

    def prepare(self):
        for setting_type in self.setting_types:
            if not getattr(self, setting_type):
                if hasattr(self, 'default_' + setting_type):
                    default = getattr(self, 'default_' + setting_type)
                    setattr(self, setting_type, getattr(self, default))
        return self