# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
from ModelHelper import ModelHelper
import numpy as np

model_helper = ModelHelper()

iris_classifier_pipeline = model_helper.get_model()

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api')
def predict():
    sepal_length = request.args.get('sepal_length')
    sepal_width = request.args.get('sepal_width')
    petal_length = request.args.get('petal_length')
    petal_width = request.args.get('petal_width')
    prediction = iris_classifier_pipeline.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    return 'The predicted value is {}. Its label is {}.'.format(str(prediction),model_helper.get_label(prediction))

@app.route('/predict', methods=['POST'])
def predict_template():
    int_features = [float(x) for x in request.form.values()]
    final_prediction = [np.array(int_features)]
    print(final_prediction)
    prediction = iris_classifier_pipeline.predict(final_prediction) # We need to pass data in 2D
    return render_template('index.html',prediction_text='The predicted value is {}. Its label is {}.'
                           .format(str(prediction),model_helper.get_label(prediction)))
    
 
if __name__ == '__main__':
    app.run(port=5010)
    #app.run(host='0.0.0.0', port=5010)