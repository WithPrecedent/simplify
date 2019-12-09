"""
.. module:: convert
:synopsis: converts external data into a usable form
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
import os

from simplify.creator.typesetter import FarmerTechnique


@dataclass
class Convert(FarmerTechnique):
    """Converts data to a usable form.

    Args:
        step(str): name of step.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    step: object = None
    parameters: object = None
    name: str = 'converter'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def _make_path(self, file_name):
        file_path = os.path.join(self.filer.external, file_name)
        return file_path

    def publish(self):
        self.file_path_in = self.make_path(self.file_in)
        self.file_path_out = self.make_path(self.file_out)
        return self

    def publish(self, ingredients):
        converted = self.method(file_path = self.file_path_in)
        self.filer.save_df(converted, file_path = self.file_path_out)
        return self