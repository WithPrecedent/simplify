""" 
LogitEncoder is a class that encodes levels of categorical variables using 
logistic regression predicted probabilities at each category level. This allows 
the user to convert high-cardinality features, which are problematic for many 
machine learning models into continuous features.

LogitEncoder is computationally expensive but may increase predictive 
accuracy versus weight of evidence and target encoding because it controls 
for other features in assigning an encoded value to each category level.

As with any categorical encoder that incorporates the predicted label, it is 
important that LogitEncoder be fitted with an isolated training dataset to 
prevent data leakage. 
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import category_encoders as ce

@dataclass
class LogitEncoder(object):
    
    """ 
    Parameters:
        
    cols: list 
    A list of columns to encode.
    
    first_encoder: str
    To apply the logistic regression encoding, the data needs to first 
    be converted from categorical to a usable format. This parameter
    sets which of the encoders to use from the category_encoders package.

    threshold: float
    A value which sets the percentage of instances for a category to appear
    in a single column from the cols list in order to be encoded. If the
    percentage of instances is below the threshold setting, the data will be
    encoded with other missing data or inadequately represented categories.        

    """
    cols : object = None
    first_encoder : str = 'helmert'
    threshold : float = 0.0

    def __post_init__(self):
        self._set_first_encoder()
        return self
       
    def _set_first_encoder(self):
        if self.first_encoder == 'backward':
            self.first = ce.BackwardDifferenceEncoder(cols = self.cols)
        if self.first_encoder == 'binary':
            self.first = ce.BinaryEncoder(cols = self.cols)
        if self.first_encoder == 'sum':
            self.first = ce.SumEncoder(cols = self.cols)
        if self.first_encoder == 'helmert':
            self.first = ce.HelmertEncoder(cols = self.cols) 
        return self   
    
    def _find_cat_cols(self, x):
        cat_cols = []
        for col in x.columns:
            if x[col].dtype == 'category':
                cat_cols.append(col)
            elif x[col].dtype == str:
                x[col] = x[col].astype('category')
                cat_cols.append(col)
        self.cols = cat_cols
        return self
  
    def _convert_below_threshold(self, x):
        for col in self.cols:   
            x['value_freq'] = x[col].value_counts() / len(x[col])
            x[col] = np.where(x['value_freq'] < self.threshold, '', x[col]) 
        x.drop('value_freq', 
               axis = 'columns', 
               inplace = True)     
        return x
   
    def fit(self, x, y, **kwargs):
        if not self.cols:
            self._find_cat_cols(x)
        if self.threshold > 0:
            x = self._convert_below_threshold(x)
        x = self.first.fit_transform(x, y)
        self.fit_logit = LogisticRegression(**kwargs).fit(x, y)
        return self
    
    def transform(self, x, y = None, **kwargs):
        x['predict_prob'] = self.fit_logit.predict_proba(x)
        coefs = pd.DataFrame(list(zip(x.columns, self.fit_logit.coef_)))
        for col in self.cols:
            coef_dict = coefs.to_dict()
            self.x[col] = coef_dict.get(self.x[col])
        return x
        
    def fit_transform(self, x, y, **kwargs):        
        x = self.fit(x, y)
        x = self.transform(x, y, **kwargs)
        return x
