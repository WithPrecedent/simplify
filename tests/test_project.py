"""
.. module:: project test
:synopsis: tests Project class
:author: Corey Rayburn Yung
:copyright: 2019
:license: Apache-2.0
"""

import pandas as pd
import pytest

from simplify.core.project import Project


def test_project():
    project = Project(
        idea = 'idea_settings.ini',
        auto_publish = False)
    print(project.iterable)
    assert project.iterable == [
        ('chef', 'draft'),
        ('chef', 'publish'),
        ('chef', 'apply'),
        ('critic', 'draft'),
        ('critic', 'publish'),
        ('critic', 'apply')]
    return

if __name__ == '__main__':
    test_project()