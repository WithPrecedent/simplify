import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer

from simplify.cookbook import Cookbook
from simplify.data import Data
from simplify.settings import Settings

cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns = np.append(cancer['feature_names'], ['target']))

settings = Settings(file_path = 'cancer_settings.ini')
data = Data(settings = settings, df = df)
data.change_column_type(columns = ['target'], data_type = bool)
data.smart_fillna()

cookbook = Cookbook(data = data, settings = settings)
cookbook.create()
cookbook.iterate()
cookbook.save_everything()
cookbook.print_best()
data.save(file_name = 'cancer_df')