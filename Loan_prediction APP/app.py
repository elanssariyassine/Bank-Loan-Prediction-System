import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_vf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    # Convert the prediction to "NO" for 0 and "YES" for 1
    output = "Can" if prediction[0] == 1 else "Can't"

    return render_template('index.html', prediction_text=f'This person {output} obtain a loan from the bank')




if __name__ == "__main__":
    app.run(debug=True)