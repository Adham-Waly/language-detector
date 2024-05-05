import os
import joblib
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data(data_dir):
    texts = []
    labels = []
    try:
        languages = os.listdir(data_dir)
    except PermissionError as e:
        print(f"Permission error: {e}")
        return texts, labels

    for lang in languages:
        lang_dir = os.path.join(data_dir, lang)
        try:
            files = os.listdir(lang_dir)
        except PermissionError as e:
            print(f"Permission error: {e}")
            continue

        for file_name in files:
            file_path = os.path.join(lang_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    texts.append(text)
                    labels.append(lang)
            except PermissionError as e:
                print(f"Permission error: {e}")
                continue

    return texts, labels


def train_model(data_dir):
    # Load data
    texts, labels = load_data(data_dir)

    if not texts or not labels:
        print("Error: No data found.")
        return None, None

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Feature extraction using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = svm_model.predict(X_test_tfidf)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    return svm_model, tfidf_vectorizer


logging.basicConfig(level=logging.INFO)


def save_model(trained_model, tfidf_vectorizer, model_dir):
    try:
        os.makedirs(model_dir, exist_ok=True)  # Create model directory if it doesn't exist
        model_path = os.path.join(model_dir, 'svm_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

        joblib.dump(trained_model, model_path)  # Save trained model
        joblib.dump(tfidf_vectorizer, vectorizer_path)  # Save TF-IDF vectorizer
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")


if __name__ == "__main__":
    data_dir = r"A:\pythonProject\langue detection\data\processed_data"
    model_dir = r"A:\pythonProject\langue detection\models"

    # Train model
    trained_model, tfidf_vectorizer = train_model(data_dir)

    if trained_model and tfidf_vectorizer:
        # Save model
        save_model(trained_model, tfidf_vectorizer, model_dir)

