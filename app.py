# app.py
from flask import Flask, render_template, request
from source.postpredictions.pipelines.prediction_pipeline import PredictPipeline, CustomData
import logging
import os
from source.postpredictions.exception import CustomException

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained model
predict_pipeline = PredictPipeline()
predict_pipeline.load_model(r"E:\endtoendpost\artifacts")

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/form")
def show_form():
    return render_template("form.html")

@app.route("/analyze", methods=["POST"])
def analyze_post():
    if request.method == "POST":
        facebook_post = request.form.get("facebookPost")

        # Assuming you don't have Emotion and User ID values
        custom_data = CustomData(facebook_post, "", "")
        
        try:
            prediction = predict_pipeline.predict(custom_data)
            logging.info(f"Facebook Post: {facebook_post}, Prediction: {prediction}")

            # Modify this part according to your requirements
            result_text = "Positive" if prediction == 1 else "Negative"
            
            return render_template("result.html", result=result_text)

        except CustomException as ce:
            logging.error(f"Error analyzing post: {ce}")
            return render_template("error.html", error_message=str(ce))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
