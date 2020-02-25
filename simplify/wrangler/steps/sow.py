"""
.. module:: sow
:synopsis: acquires and does basic preparation of data
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from dataclasses import dataclass

from simplify.core.definitionsetter import WranglerTechnique


"""DEFAULT_OPTIONS are declared at the top of a module with a SimpleDirector
subclass because siMpLify uses a lazy importing system. This locates the
potential module importations in roughly the same place as normal module-level
import commands. A SimpleDirector subclass will, by default, add the
DEFAULT_OPTIONS to the subclass as the 'options' attribute. If a user wants
to use another set of 'options' for a subclass, they just need to pass
'options' when the class is instanced.
"""
DEFAULT_OPTIONS = {
    'download': ['simplify.wrangler.steps.download', 'Download'],
    'scrape': ['simplify.wrangler.steps.scrape', 'Scrape'],
    'convert': ['simplify.wrangler.steps.convert', 'Convert'],
    'divide': ['simplify.wrangler.steps.divide', 'Divide']}


@dataclass
class Sow(SimpleIterable):
    """Acquires and performs basic preparation of data sources.

    Args:
        steps(dict): dictionary containing keys of WranglerTechnique names (strings)
            and values of WranglerTechnique class instances.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    steps: object = None
    name: str = 'sower'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def draft(self) -> None:
        self.needed_parameters = {'convert': ['file_in', 'file_out',
                                                 'method'],
                                  'download': ['file_url', 'file_name'],
                                  'scrape': ['file_url', 'file_name'],
                                  'split': ['in_folder', 'out_folder',
                                                'method']}
        if self.step in ['split']:
            self.import_folder = 'raw'
            self.export_folder = 'interim'
        else:
            self.import_folder = 'external'
            self.export_folder = 'external'
        return self

    def publish(self, dataset):
        self.algorithm.implement(dataset)
        return dataset
