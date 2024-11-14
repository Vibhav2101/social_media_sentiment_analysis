from flask import Flask, request, render_template
import pickle
from utils import preprocess_text
import plotly.graph_objects as go
import plotly.utils
import json

app = Flask(__name__)

# Load model and vectorizer
with open('models/sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('input_text')
    if not text:
        return "Error: No input text provided", 400

    # Process text, get prediction and probability
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction_proba = model.predict_proba(vectorized_text)[0]
    prediction = model.predict(vectorized_text)[0]
    probability = round(float(prediction_proba.max()) * 100, 2)

    # Set sentiment emoji based on prediction
    if prediction == "Positive":
        sentiment_emoji = "😃"
    elif prediction == "Neutral":
        sentiment_emoji = "😐"
    elif prediction == "Negative":
        sentiment_emoji = "😞"
    else:
        sentiment_emoji = "❓"  # Fallback emoji if prediction label is unexpected

    # Create Plotly gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={"text": "Confidence Level"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "green" if prediction == "Positive" else "gray" if prediction == "Neutral" else "red"},
        }
    ))
    gauge_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Render template with prediction results
    return render_template(
        'index.html',
        prediction=prediction,
        confidence=probability,
        gauge_json=gauge_json,
        sentiment_emoji=sentiment_emoji
    )

if __name__ == "__main__":
    app.run(debug=True)
