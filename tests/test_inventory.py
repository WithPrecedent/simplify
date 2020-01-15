"""
.. module:: inventory test
:synopsis: tests Idea class
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

from pathlib import Path
from simplify.core.idea import Idea
from simplify.core.inventory import Inventory


def test_inventory():
    idea = Idea(
        configuration = Path.cwd().joinpath('tests', 'idea_settings.ini'))
    inventory = Inventory(idea = idea)
    assert inventory.folders['root'] == Path.cwd().joinpath('..\..')
    return


if __name__ == '__main__':
    test_inventory()