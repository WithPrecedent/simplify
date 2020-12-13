"""
.. module:: clerk test
:synopsis: tests Idea class
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from pathlib import pathlib.Path

from simplify.core.idea import Idea
from simplify.core.clerk import Clerk


def test_clerk():
    idea = Idea(
        configuration = pathlib.Path.cwd().joinpath('tests', 'idea_settings.ini'))
    clerk = Clerk(idea = idea)
    assert clerk.folders['root'] == pathlib.Path.cwd().joinpath('..\..')
    return


if __name__ == '__main__':
    test_clerk()