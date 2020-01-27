"""
.. module:: project test
:synopsis: tests Project class
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from pathlib import Path

import pandas as pd
import pytest

from simplify.core.idea import Idea
from simplify.core.project import Project


def test_project():
    idea = Idea(
        configuration = Path.cwd().joinpath('tests', 'idea_settings.ini'))
    project = Project(idea = idea)
    print('test', project.library)
    return

if __class__.__name__ == '__main__':
    test_project()