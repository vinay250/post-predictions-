from flask import Flask, render_template, request
from source.postpredictions.pipelines.prediction_pipeline import PredictPipeline, CustomData
import logging
import os

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained model
predict_pipeline = PredictPipeline.load_model(r"E:\endtoendpost\artifacts")


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

        # Perform your analysis using the trained model
        custom_data = CustomData(facebook_post, "", "")  # Assuming you don't have Emotion and User ID for prediction
        prediction = predict_pipeline.predict(custom_data)

        logging.info(f"Facebook Post: {facebook_post}, Prediction: {prediction}")

        return render_template("result.html", result=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
