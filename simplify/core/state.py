"""
.. module:: state
:synopsis: siMpLify state machine
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify import SimpleClass


@dataclass
class StateRegulator(SimpleClass):

    initial_state : str = ''
    overwrite : bool = True

    def __post_init__(self):
        super().__post_init__()
        self.state = self.options['initial_state']
        self._set_initial_state()
        return self

    def _set_initial_state(self):
        if self.initial_state in self.options:
            self.state = self.options[self.initial_state]
        elif 'farmer' in self.subpackages:
            setting = self.listify(
                    self.idea.configuration['almanac']['almanac_steps'])[0]
            self.state = self.options[setting]
        elif 'chef' in self.subpackages:
            self.state = self.options['unsplit']
        elif 'review' in self.subpackages:
            self.state = self.options['review']
        elif 'review' in self.subpackages:
            self.state = self.options['canvas']
        else:
            self.state = self.options['unsplit']
        return self

    def draft(self):
        self.options = {
                'sow' : SowState,
                'harvest' : HarvestState,
                'clean' : CleanState,
                'bale' : BaleState,
                'deliver' : DeliverState,
                'unsplit' : UnsplitState,
                'split' : SplitState,
                'review' : ReviewState,
                'canvas' : CanvasState}
        return self

    def advance(self, stage):
        self.state = self.state.advance(stage)
        for option, setting in self.files.items():
            if self.overwrite:
                setattr(self, option, setting)
        return self

@dataclass
class State(object):

    def __post_init__(self):
        super().__post_init__()
        return self

    def advance(self, stage):
        pass
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.__class__.__name__

@dataclass
class SowState(State):

    def __post_init__(self):
        self.files = {
                'folder_in' : 'raw',
                'folder_out' : 'raw',
                'file_in' : None,
                'file_out' : None,
                'format_in' : 'txt',
                'format_out' : 'txt'}
        return self

@dataclass
class HarvestState(State):

    def __post_init__(self):
        return self

    def advance(self):
        self.files = {
                'folder_in' : 'raw',
                'folder_out' : 'interim',
                'file_in' : None,
                'file_out' : 'harvested_data',
                'format_in' : 'txt',
                'format_out' : 'csv'}
        self.datatypes = {'list' : 'list',
                          'pattern' : 'pattern'}
        return self

@dataclass
class CleanState(State):

    def __post_init__(self):
        return self

@dataclass
class BaleState(State):

    def __post_init__(self):
        return self

@dataclass
class DeliverState(State):

    def __post_init__(self):
    def advance(self):
        self.files = {
                'folder_in' : 'raw',
                'folder_out' : 'interim',
                'file_in' : 'sowed_data',
                'file_out' : 'harvested_data',
                'format_in' : 'txt',
                'format_out' : 'csv'}
        self.datatypes = {'list' : 'list',
                          'pattern' : 'pattern'}
        return self


@dataclass
class UnsplitState(State):

    def __post_init__(State):
        return self

@dataclass
class SplitState(object):

    def __post_init__(State):
        return self

@dataclass
class ReviewState(object):

    def __post_init__(State):
        return self


@dataclass
class CanvasState(object):

    def __post_init__(self):
        return self
