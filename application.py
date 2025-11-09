import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exception import CustomException

# No more webbrowser, Timer, or os imports needed

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    # Renders the main page
    return render_template('index.html')

# Route for handling the prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # If someone just visits /predictdata, show them the form
        return render_template('index.html')
    else:
        # This block runs when the user 'POSTS' (submits) the form
        
        logging.info("Form data received for prediction.")
        
        try:
            # --- This is the 'Pylance-friendly' way to get form data ---
            reading_score_str = request.form.get('reading_score')
            writing_score_str = request.form.get('writing_score')

            reading_score_int = int(reading_score_str) if reading_score_str else 0
            writing_score_int = int(writing_score_str) if writing_score_str else 0
            
            # 1. Get data from the form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                
                # Pass the guaranteed-to-be-int variables
                reading_score=reading_score_int,
                writing_score=writing_score_int
            )
            
            # 2. Convert form data to a DataFrame
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Data for prediction: \n{pred_df.to_string()}")

            # 3. Get a prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            # Format the result to two decimal places
            output = f"{results[0]:.2f}"
            logging.info(f"Prediction result: {output}")
            
            # 4. Display the result on the same page
            return render_template('index.html', results=output)
        
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            # Pass the error message back to the HTML page
            return render_template('index.html', error=str(e))

# This block is for local testing ONLY
# AWS Elastic Beanstalk will NOT run this
if __name__ == "__main__":
    # Run the app in production mode (debug=False)
    app.run(host="0.0.0.0", port=5000, debug=False)