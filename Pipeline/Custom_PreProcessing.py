from sklearn.base import BaseEstimator, TransformerMixin

class Custom_PreProcessing(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for our use-case
    """
    def __init__(self):
        pass
    
    def transform(self, df):
        """Regular transform() that is a help for training, validation & testing datasets
           (NOTE: The operations performed here are the ones that we did prior to this cell)
        """
        pred_var = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome',
                    'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
        df = df[pred_var]
        
        # Fill Missing Value object datatype
        df['Dependents'] = df['Dependents'].fillna(0)
        df['Married'] = df['Married'].fillna('No')
        df['Gender'] = df['Gender'].fillna(df['Gender'].value_counts().sort_values(ascending=False).index[0])
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(self.term_mean_)
        df['LoanAmount'] = df['LoanAmount'].fillna(self.amt_mean_)
        df['Self_Employed'] = df['Self_Employed'].fillna('No')
        df['Credit_History'] = df['Credit_History'].fillna(1)
        
        gender = {'Female':0,'Male':1}
        married = {'No':0,'Yes':1}
        education = {'Graduate':0,'Not Graduate':1}
        self_employed = {'No':0,'Yes':1}
        property_area = {'Urban':0,'Rural':1,'Semiurban':2}
        
        df.replace({'Gender':gender, 'Married':married, 'Education':education
                    , 'Self_Employed':self_employed, 'Property_Area':property_area})

        return df

        
    def fit(self, df, y=None, **fit_params):
        """Fitting the Training dataset & calculating the required values from train
           e.g: We will need the mean of X_train['Loan_Amount_Term'] that will be used in
                transformation of X_test
        """
        
        self.term_mean_ = df['Loan_Amount_Term'].mean()
        self.amt_mean_ = df['LoanAmount'].mean()
        return self