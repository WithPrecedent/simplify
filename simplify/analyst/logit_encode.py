""" 
LogitEncoder is a class built on a scikit-learn transformer that encodes 
levels of categorical variables using logistic regression coefficients or 
marginal effects at each category level. This allows the user to convert high-
cardinality features, which are problematic for many machine learning
models (particularly tree methods) into continuous features.

LogitEncoder is computationally expensive but may increase predictive 
accuracy versus weight of evidence and target encoding because it controls 
for other features in assigning an encoded value to each category level.

As with any categorical encoder that incorporates the predicted label, it is 
important that LogitEncoder be applied to an isolated dataset to prevent data
leakage. 

"""
#%%
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.linear_model as lm

import category_encoders as ce

#%%
class LogitEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, encode_cols = None, first_encoder = 'helmert',
                 col_prefix = 'coef_', verbose = False, threshold = 30, 
                 below_threshold = 'mean', encoded_values = 'coefficients'):

        """ Parameters:
            
        encode_cols: list 
        A list of columns to encode.
        
        first_encoder: str
        To apply the logistic regression encoding, the data needs to first 
        be converted from categorical to a usable format. This parameter
        sets which of the encoders to use from the category_encoders package.
        
        new_col: str
        If set to a string, a new column, with the string as the header,
        will be created to store the encoded values.
        
        verbose: bool 
        A value of True will include progress updates. verbose of False 
        will not include any progress updates.
    
        threshold: int
        A value which sets the number of instances for a value to appear
        in a single column from the encode_cols list in order to be 
        encoded. If there are the number of instances is equal to or 
        below the threshold setting, an encoded value will not be used.
            
        below_threshold: str
        If threshold is greater than 0 then below_threshold sets what 
        value should be used in place of a logistic-regression-based 
        value for that particular instance. If set to 'mean', the mean
        of the column will be used. If set to 'zero', the value of 0 
        will be used.
            
        encoded_values: str
        If values is set to 'coefficients', the logistic regression 
        coefficient for each categorical level shall be encoded.
        If values is set to 'effects', the estimated marginal effect
        of each categorical level shall be encoded. Selecting 'effects'
        is substantially more computationally expensive than using 
        coefficients.
        """
        self.cols = encode_cols
        self.first = first_encoder
        self.col_prefix = col_prefix
        self.verbose = verbose
        self.threshold = threshold
        self.below_threshold = below_threshold
        self.encoded_values = encoded_values
        self.X = pd.DataFrame
        self.y = pd.Series
        self.feature_names = None       
        return
   
    def fit_transform(self, X, y, **kwargs):
        """ To fit LogitEncoder, pass features, X, and label, y with any 
        arguments accepted by scikit-learn LogisticRegression
        """
        self.X = X
        self.y = y
        
        """ If a single value or no value is passed for encoded_cols, generate
        a list of either all columns or the single listed column.
        """
        if self.cols is None:
            self._cat_cols()
        elif isinstance(self.cols, str):
            self.cols = self.cols.tolist()
            
        self._set_first_encoder()
        self._logit_encode()
        
        if self.threshold > 0:
            self._cutoff()
        return self.X

    def _cat_cols(self):
        cat_cols = []
        for col in self.cols:
            if pd.is_category(self.X[col]):
                cat_cols.append(col)
        return cat_cols
    
    def _set_first_encoder(self):
        if self.first == 'backward':
            self.first_encoder = ce.BackwardDifferenceEncoder(cols = self.cols)
        if self.first == 'binary':
            self.first_encoder= ce.BinaryEncoder(cols = self.cols)
        if self.first == 'sum':
            self.first_encoder = ce.SumEncoder(cols = self.cols)
        if self.first == 'helmert':
            self.first_encoder = ce.HelmertEncoder(cols = self.cols) 
        return self   
    
    def _logit_encode(self, **kwargs):
        self.first_encoder.fit_transform(self.X, self.y)
        print(self.X.head())
        self.logit = lm.LogisticRegression()
        self.logit.fit(self.X, self.y)
        for col in self.cols:
            col_name = self.col_prefix + col
            self.coefs = pd.DataFrame(list(zip(self.X.columns, 
                                               self.logit.coef_)))
            self.coef_dict = self.coefs.to_dict()
            self.X[col_name] = self.coef_dict.get(self.X[col])
#        self.X.drop(self.cols, axis = 1, inplace = True)
        return self
  
    def _cutoff(self):           
        for col in self.cols:
            col_name = self.col_prefix + col
            if self.below_threshold == 'zero':
                substitute_value = 0
            elif self.below_threshold == 'mean':
                substitute_value = np.mean(self.X[self.col_name])
            value_counts = self.X[col].value_counts()
            self.X[col_name] = np.where(value_counts <= self.threshold,
                                   substitute_value,
                                   self.X[col_name])            
        return self

