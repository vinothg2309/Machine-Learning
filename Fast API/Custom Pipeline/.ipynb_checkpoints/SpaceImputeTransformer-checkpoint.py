import numpy as np
class SpaceImputeTransformer():
    def __init__(self):
        pass # Assigning function passed to it
    
    def fit(self,X,y=None, **fit_params):
        print('Inside SpaceImputeTransformer fit')
        return self
    
    def transform(self,input_df, **transform_params):
        print('Inside SpaceImputeTransformer transform')
        for col in input_df.columns:
            blank_values = input_df[col].replace(r'^\s*$', np.nan,regex=True).isna().sum()
            #print('Col : ', col , ' blank_values : ', blank_values)
            if(blank_values > 0):
                input_df[col] = input_df[col].replace(r'^\s*$', np.nan,regex=True);
        return input_df