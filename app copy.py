from flask import Flask, render_template, request
import numpy as np
from src.utils import load_object
import pickle
import os

app = Flask(__name__)
full_model_path = os.path.join("artifacts", "model.pkl")

#model = pickle.load(open('model.pkl', 'rb'))
model = load_object(file_path=full_model_path)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = float(request.form['City'])
    val2 = float(request.form['Day'])
    val3 = float(request.form['Room Type'])
    val4 = request.form['Shared Room']
    val5 = request.form['Private Room']
    val6 = request.form['Person Capacity']
    val7 = request.form['Superhost']
    val8 = request.form['Multiple Rooms']
    val9 = request.form['Business']
    val10 = request.form['Cleanliness Rating']
    val11 = request.form['Guest Satisfaction']
    val12 = request.form['Bedrooms']
    val13 = request.form['City Center (km)']
    val14 = request.form['Metro Distance (km)']
    val15 = request.form['Attraction Index']
    val16 = request.form['Normalised Attraction Index']
    val17 = request.form['Restraunt Index']
    val18 = request.form['Normalised Restraunt Index']
    arr = np.array([val1, val2, val3, val4, val5, val6, val7, val8, val9,
                   val10, val11, val12, val13, val14, val15, val16, val17, val18])
    
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('index.html', data=int(pred))


if __name__ == '__main__':
#    app.run(debug=True)
	 app.run(host="0.0.0.0",port=9090)


