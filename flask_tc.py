from flask import Flask, request, jsonify, render_template_string
import joblib

# Load saved model and vectorizer
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Report Category Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 50px; }
        h1 { color: #333; }
        form { margin-bottom: 20px; }
        input[type=text] { width: 400px; padding: 10px; margin: 10px 0; }
        input[type=submit] { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        input[type=submit]:hover { background-color: #45a049; }
        .result { margin-top: 20px; font-size: 18px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>📊 Report Category Classification</h1>
    <form method="post" action="/predict_form">
        <input type="text" name="query" placeholder="Enter your query here" required>
        <br>
        <input type="submit" value="Predict">
    </form>

    {% if prediction %}
        <div class="result">
            ✅ Query: "{{ query }}" <br>
            👉 Predicted Category: <span style="color:blue">{{ prediction }}</span>
        </div>
    {% endif %}
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict_form", methods=["POST"])
def predict_form():
    query = request.form["query"]
    query_tfidf = vectorizer.transform([query])
    prediction = model.predict(query_tfidf)[0]

    return render_template_string(HTML_TEMPLATE, prediction=prediction, query=query)

# API endpoint (still available for Postman/curl use)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    query = data.get("text", "")
    query_tfidf = vectorizer.transform([query])
    prediction = model.predict(query_tfidf)[0]
    return jsonify({"query": query, "predicted_category": prediction})

if __name__ == "__main__":
    app.run(debug=True)
