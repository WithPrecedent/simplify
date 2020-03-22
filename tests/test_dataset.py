"""
.. module:: test dataset
:synopsis: tests Dataset class
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from pathlib import pathlib.Path

import pandas as pd

from simplify.core.dataset import Dataset


def test_dataset():
    raw_data = [['allison', 25], ['brian', 30], ['corey', 40]]
    df = pd.DataFrame(data = pd.DataFrame(
        raw_data,
        columns = ['name', 'age'],
        index = None))
    data = Dataset.create(data = df)
    data.downcast()
    assert data['name'].tolist() == ['allison', 'brian', 'corey']
    assert len(data) == 3
    data.create_xy(label = 'age')
    assert data.x['name'].tolist() == ['allison', 'brian', 'corey']
    return


if __name__ == '__main__':
    test_dataset()