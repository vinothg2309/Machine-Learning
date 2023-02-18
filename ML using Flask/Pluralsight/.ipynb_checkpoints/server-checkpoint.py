import numpy as np
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
print(app)

model = joblib.load(open('models/pipe_clf_checkpoint.joblib', 'rb'))
print(model)
model_clf = model['pipeline_clf']

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model_clf.predict(data['x'])
    output_text = "Text: "+str(data['x'])
    output = "Class: "+ str(prediction)
    return jsonify(output_text, output)

if __name__ == '__main__':
    app.run(port=9000, debug=True)
