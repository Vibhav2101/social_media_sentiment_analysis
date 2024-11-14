def predict_naive_bayes(text, model, vectorizer):
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    return prediction[0]

def calculate_probability(text, model, vectorizer):
    transformed_text = vectorizer.transform([text])
    class_probs = model.predict_proba(transformed_text)[0]
    labels = model.classes_
    return dict(zip(labels, class_probs))
