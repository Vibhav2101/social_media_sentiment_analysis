# Step 1: Import necessary libraries
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 2: Import the preprocess_text function from utils.py
from utils import preprocess_text  # Ensure that utils.py is in the same directory

# Step 3: Load the updated dataset
df = pd.read_csv("data/dataset.csv")

# Step 4: Preprocess the text
df['processed_text'] = df['text'].apply(preprocess_text)

# Step 5: Split the data into features and labels
X = df['processed_text']
y = df['sentiment']

# Step 6: Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Step 7: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Step 8: Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 9: Save the trained model and vectorizer
with open('models/sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('models/tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and TF-IDF vectorizer have been saved.")
