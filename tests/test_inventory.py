"""
.. module:: filer test
:synopsis: tests Idea class
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from pathlib import Path

from simplify.core.idea import Idea
from simplify.core.filer import Filer


def test_filer():
    idea = Idea(
        configuration = Path.cwd().joinpath('tests', 'idea_settings.ini'))
    filer = Filer(idea = idea)
    assert filer.folders['root'] == Path.cwd().joinpath('..\..')
    return


if __name__ == '__main__':
    test_filer()