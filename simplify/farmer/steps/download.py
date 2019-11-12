"""
.. module:: download
:synopsis: acquires data from online source
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from dataclasses import dataclass
import os
import requests

from simplify.core.technique import FarmerTechnique


@dataclass
class Download(FarmerTechnique):
    """Acquires data from an online source.

    Args:
        technique(str): name of technique.
        parameters(dict): dictionary of parameters to pass to selected
            algorithm.
        name(str): name of class for matching settings in the Idea instance
            and elsewhere in the siMpLify package.
        auto_draft(bool): whether 'publish' method should be called when
            the class is instanced. This should generally be set to True.
    """

    technique: object = None
    parameters: object = None
    name: str = 'downloader'
    auto_draft: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        return self

    def publish(self):
        return self

    def publish(self, ingredients):
        """Downloads file from a URL if the file is available."""
        file_path = os.path.join(self.depot.external,
                                 self.file_name)
        file_response = requests.get(self.file_url)
        with open(file_path, 'wb') as file:
            file.write(file_response.content)
        return self

