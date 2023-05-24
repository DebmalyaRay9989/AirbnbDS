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
                        City=request.form.get('City'),
                        Day=request.form.get('Day'),
                        room_type=request.form.get('room_type'),
                        shared_room = request.form.get('shared_room'),
                        private_room = request.form.get('private_room'),
                        person_capacity = request.form.get('person_capacity'),
                        superhost = request.form.get('superhost'),
                        multiple_rooms = request.form.get('multiple_rooms'),
                        business = request.form.get('business'),
                        cleanliness_rating = request.form.get('cleanliness_rating'),
                        guest_satisfaction = request.form.get('guest_satisfaction'),
                        bedrooms = request.form.get('bedrooms'),
                        city_center_km = request.form.get('city_center_km'),
                        metro_distance_km = request.form.get('metro_distance_km'),
                        attraction_index = request.form.get('attraction_index'),
                        normalised_attraction_index = request.form.get('normalised_attraction_index'),
                        restraunt_index = request.form.get('restraunt_index'),
                        normalised_restraunt_index = request.form.get('normalised_restraunt_index')
        
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        #return render_template('home.html', results = results[0])
        return render_template('home.html', results = float(results[0]))


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 9090)
    # app.run(debug=True)