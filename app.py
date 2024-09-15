from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model and data
model = joblib.load('DT_Model.pkl')
data = pd.read_csv('Data.csv')

# Extract unique values for dropdowns
airline_options = data['airline'].unique()
source_city_options = data['source_city'].unique()
destination_city_options = data['destination_city'].unique()
class_options = data['class'].unique()
departure_time_options = data['departure_time'].unique()
arrival_time_options = data['arrival_time'].unique()

#label encoders
label_encoders = {}
for column in ['airline', 'source_city', 'destination_city', 'class', 'departure_time', 'arrival_time']:
    le = LabelEncoder()
    le.fit(data[column])
    label_encoders[column] = le

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        form_data = {
            'airline': request.form['airline'],
            'source_city': request.form['source_city'],
            'departure_time': request.form['departure_time'],
            'stops': int(request.form['stops']),
            'arrival_time': request.form['arrival_time'],
            'destination_city': request.form['destination_city'],
            'class': request.form['class'],
            'duration': float(request.form['duration']),
            'days_left': int(request.form['days_left'])
        }

        # Create a DataFrame for input
        input_df = pd.DataFrame([form_data])

        # Apply label encoding to the input
        for column, le in label_encoders.items():
            input_df[column] = le.transform(input_df[column])

        # Prepare the input array for prediction
        input_array = input_df.values

        # Make prediction
        try:
            prediction = model.predict(input_array)
            return render_template('predict.html', prediction=prediction[0],
                                  form_data=form_data,
                                  airline_options=airline_options,
                                  source_city_options=source_city_options,
                                  destination_city_options=destination_city_options,
                                  class_options=class_options,
                                  departure_time_options=departure_time_options,
                                  arrival_time_options=arrival_time_options)
        except Exception as e:
            return f'An error occurred: {e}'

    return render_template('predict.html',
                          form_data=None,
                          airline_options=airline_options,
                          source_city_options=source_city_options,
                          destination_city_options=destination_city_options,
                          class_options=class_options,
                          departure_time_options=departure_time_options,
                          arrival_time_options=arrival_time_options,
                          prediction=None)
@app.route('/video')
def video():
    return render_template('comingsoon.html')
if __name__ == '__main__':
    app.run(debug=True)
