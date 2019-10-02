"""
.. module:: state
:synopsis: siMpLify state machine
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass


@dataclass
class Machine(object):

    idea: object
    initial_state: str = ''
    overwrite: bool = True

    def __post_init__(self):
        self.draft()
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

    def _set_state(self):
        if self.name in self.options:
            self.state = self.name
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
