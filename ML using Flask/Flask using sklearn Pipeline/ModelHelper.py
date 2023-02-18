# -*- coding: utf-8 -*-

from joblib import load

class ModelHelper:
    def get_model(self):
        return load('iris_pipeline_classification.joblib')
    
    def get_label(self,prediction):
        actual_labels = ['setosa', 'versicolor', 'virginica']
        return actual_labels[prediction[0]]