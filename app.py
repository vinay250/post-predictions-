from source.postpredictions.pipelines.prediction_pipeline import CustomData,PredictPipeline
from flask import Flask, render_template, request
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

        # Perform your analysis here with the provided Facebook post

        # For demonstration purposes, let's assume the result is "Positive"
        result = "Positive"

        logging.info(f"Facebook Post: {facebook_post}, Result: {result}")

        return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
