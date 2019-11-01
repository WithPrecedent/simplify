"""
.. module:: harvest
:synopsis: parses data sources to create pandas DataFrame
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
import os

from simplify.core.technique import FarmerTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleClass
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleClass subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'organize': ['simplify.core.retool', 'ReTool'],
    'parse': ['simplify.core.retool', 'ReTool']}


@dataclass
class Harvest(SimpleIterable):
    """Extracts data from text or other sources.

    Args:
        techniques(dict): dictionary containing keys of FarmerTechnique names (strings)
            and values of FarmerTechnique class instances.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    techniques: object = None
    name: str = 'harvester'
    auto_draft: bool = True

    def __post_init__(self):
        super().__post_init__()
        return self

    def _publish_organize(self, key):
        file_path = os.path.join(self.depot.instructions,
                                 'organizer_' + key + '.csv')
        self.parameters = {'technique': self.technique,
                           'file_path': file_path}
        algorithm = self.options[self.technique](**self.parameters)
        self._set_columns(algorithm)
        return algorithm

    def _publish_parse(self, key):
        file_path = os.path.join(self.depot.instructions,
                                 'parser_' + key + '.csv')
        self.parameters = {'technique': self.technique,
                           'file_path': file_path}
        algorithm = self.options[self.technique](**self.parameters)
        return algorithm

    def draft(self):
        return self

    def _set_columns(self, algorithm):
        prefix = algorithm.matcher.section_prefix
        if not hasattr(self, 'columns'):
            self.columns = []
        new_columns = list(algorithm.expressions.values())
        new_columns = [prefix + '_' + column for column in self.columns]
        self.columns.extend(new_columns)
        return self

    def _implement_organize(self, ingredients, algorithm):
        ingredients.df, ingredients.source = algorithm.implement(
                df = ingredients.df, source = ingredients.source)
        return ingredients

    def _implement_parse(self, ingredients, algorithm):
        ingredients.df = algorithm.implement(df = ingredients.df,
                                         source = ingredients.source)
        return ingredients

    def publish(self):
        for key in self.parameters:
            if hasattr(self, '_publish_' + self.technique):
                algorithm = getattr(
                        self, '_publish_' + self.technique)(key = key)
            else:
                algorithm = getattr(self, '_publish_generic_list')(key = key)
            self.algorithms.append(algorithm)
        return self