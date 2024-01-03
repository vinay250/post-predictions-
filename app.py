# app.py
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

        try:
            logging.info(f"Facebook Post: {facebook_post}")

            # Directly determine sentiment based on some criteria (modify as needed)
            # For example, you can check for certain keywords in the post
            if "happy" in facebook_post.lower():
                result_text = "Positive"
            else:
                result_text = "Negative"
            
            return render_template("result.html", result=result_text)

        except Exception as e:
            logging.error(f"Error processing post: {e}")
            return render_template("error.html", error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
