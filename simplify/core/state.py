"""
.. module:: state
:synopsis: siMpLify state machine
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass


@dataclass
class StateRegulator(object):

    idea: object
    initial_state: str = ''
    overwrite: bool = True

    def __post_init__(self):
        self.draft()
        self.idea.inject(instance = self, sections = ['general', 'files'])
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
        elif 'artist' in self.subpackages:
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

    def advance(self):
        if hasattr(self, 'name') and self.name in self.options:
            self.state = self.options[self.name].advance()
        return self

@dataclass
class State(object):

    def __post_init__(self):
        super().__post_init__()
        return self

    def advance(self):
        for option, setting in self.files.items():
            if self.overwrite:
                setattr(self, option, setting)
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.__class__.__name__

@dataclass
class SowState(State):

    def __post_init__(self):
        return self
    
    def advance(self):
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
        self.module_trigger = 'clean'
        return self

    def advance(self):
        self.files = {
                'folder_in' : 'interim',
                'folder_out' : 'interim',
                'file_in' : 'harvested_data',
                'file_out' : 'cleaned_data',
                'format_in' : 'csv',
                'format_out' : 'csv'}
        self.datatypes = {'list' : 'list',
                          'pattern' : 'pattern'}
        return self

@dataclass
class BaleState(State):

    def __post_init__(self):
        return self

    def advance(self):
        self.files = {
                'folder_in' : 'interim',
                'folder_out' : 'interim',
                'file_in' : 'cleaned_data',
                'file_out' : 'baled_data',
                'format_in' : 'csv',
                'format_out' : 'csv'}
        self.datatypes = {'list' : 'list',
                          'pattern' : 'pattern'}
        return self


@dataclass
class DeliverState(State):

    def __post_init__(self):
        return self
        
    def advance(self):
        self.files = {
                'folder_in' : 'interim',
                'folder_out' : 'processed',
                'file_in' : ['cleaned_data', 'baled_data'],
                'file_out' : 'final_data',
                'format_in' : 'csv',
                'format_out' : 'csv'}
        self.datatypes = {'list' : 'category',
                          'pattern' : 'category'}
        return self


@dataclass
class UnsplitState(State):

    def __post_init__(self):
        return self

    def advance(self):
        self.files = {
                'folder_in' : 'processed',
                'folder_out' : 'processed',
                'file_in' : 'final_data',
                'file_out' : None,
                'format_in' : 'txt',
                'format_out' : 'csv'}
        self.datatypes = {'list' : 'category',
                          'pattern' : 'category'}
        return self

@dataclass
class SplitState(object):

    def __post_init__(self):
        return self

    def advance(self):
        self.files = {
                'folder_in' : 'processed',
                'folder_out' : 'processed',
                'file_in' : 'final_data',
                'file_out' : None,
                'format_in' : 'txt',
                'format_out' : 'csv'}
        self.datatypes = {'list' : 'category',
                          'pattern' : 'category'}
        return self

@dataclass
class ReviewState(object):

    def __post_init__(self):
        return self

    def advance(self):
        self.files = {
                'folder_in' : 'processed',
                'folder_out' : 'processed',
                'file_in' : 'final_data',
                'file_out' : 'predicted_data',
                'format_in' : 'txt',
                'format_out' : 'csv'}
        self.datatypes = {'list' : 'category',
                          'pattern' : 'category'}
        return self


@dataclass
class CanvasState(State):

    def __post_init__(self):
        return self

    def advance(self):
        self.files = {
                'folder_in' : 'processed',
                'folder_out' : 'processed',
                'file_in' : ['predicted_data', 'final_data'],
                'file_out' : None,
                'format_in' : 'txt',
                'format_out' : 'csv'}
        self.datatypes = {'list' : 'category',
                          'pattern' : 'category'}
        return self
