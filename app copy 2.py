from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for a home page


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
                        val1=request.form.get['City'],
                        val2=request.form.get['Day'],
                        val3=request.form.get['Room Type'],
                        val4 = float(request.form.get['Shared Room']),
                        val5 = float(request.form.get['Private Room']),
                        val6 = float(request.form.get['Person Capacity']),
                        val7 = float(request.form.get['Superhost']),
                        val8 = float(request.form.get['Multiple Rooms']),
                        val9 = float(request.form.get['Business']),
                        val10 = float(request.form.get['Cleanliness Rating']),
                        val11 = float(request.form.get['Guest Satisfaction']),
                        val12 = float(request.form.get['Bedrooms']),
                        val13 = float(request.form.get['City Center (km)']),
                        val14 = float(request.form.get['Metro Distance (km)']),
                        val15 = float(request.form.get['Attraction Index']),
                        val16 = float(request.form.get['Normalised Attraction Index']),
                        val17 = float(request.form.get['Restraunt Index']),
                        val18 = float(request.form.get['Normalised Restraunt Index'])
        
        )
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html', results = results[0])


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 9090)
    # app.run(debug=True)