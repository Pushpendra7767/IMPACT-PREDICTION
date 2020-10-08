import numpy as np
import pandas as pd
from flask import send_from_directory, Flask, request, jsonify, render_template, redirect, url_for
import csv
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('untitled4.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    hp=np.reshape(int_features,(1,13))
       
    prediction = model.predict(hp)

    output = round(prediction[0], 2)

    return render_template('untitled4.html', prediction_text='IMPACT {}'.format(output))

@app.route('/data',methods=['GET','POST'])
def data():
    if request.method =='POST':
        f=request.form['csvfile']
        m=pd.read_csv(f)
        i=m['ID']
        pred=model.predict(m)
        o=[]
        for row in pred:
            o.append(row)
        data=pd.DataFrame(list(zip(i,o)),columns=['ID','RESULT'])
        return render_template('untitled14.html',tables=[data.to_html(classes='data')], titles=data.columns.values)
    




'''
@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    For direct API calls trought request
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
'''
if __name__ == "__main__":
    app.run(debug=True)
