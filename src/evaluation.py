import os
import joblib
import logging
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)


def load_model(model_dir):
    model_path = os.path.join(model_dir, 'svm_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

    try:
        trained_model = joblib.load(model_path)
        tfidf_vectorizer = joblib.load(vectorizer_path)
        logging.info("Model loaded successfully.")
        return trained_model, tfidf_vectorizer
    except FileNotFoundError:
        logging.error("Model files not found. Please check the model directory.")
        return None, None
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None, None


def load_test_data(data_dir):
    try:
        languages = os.listdir(data_dir)
        texts = []
        labels = []
        for i, lang in enumerate(languages):
            lang_dir = os.path.join(data_dir, lang)
            files = os.listdir(lang_dir)
            for file_name in files:
                with open(os.path.join(lang_dir, file_name), 'r', encoding='utf-8') as file:
                    text = file.read()
                    texts.append(text)
                    labels.append(i)
        logging.info("Test data loaded successfully.")
        return texts, labels
    except FileNotFoundError:
        logging.error("Test data directory not found. Please check the directory path.")
        return [], []
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        return [], []


def evaluate_model(trained_model, tfidf_vectorizer, test_texts, test_labels):
    if trained_model is None or tfidf_vectorizer is None:
        logging.error("Model or vectorizer is not loaded. Evaluation aborted.")
        return

    try:
        # Transform test data using the TF-IDF vectorizer
        test_data_tfidf = tfidf_vectorizer.transform(test_texts)

        # Predictions
        y_pred = trained_model.predict(test_data_tfidf)

        # Calculate accuracy
        accuracy = accuracy_score(test_labels, y_pred)
        logging.info(f"Accuracy: {accuracy}")

        # Print classification report
        print("Classification Report:")
        print(classification_report(test_labels, y_pred))
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")


if __name__ == "__main__":
    model_dir = r"A:\pythonProject\langue detection\models"
    test_data_dir = r"A:\pythonProject\langue detection\data\processed_data\test"

    # Load model
    trained_model, tfidf_vectorizer = load_model(model_dir)

    # Load test data
    test_texts, test_labels = load_test_data(test_data_dir)

    # Evaluate model
    evaluate_model(trained_model, tfidf_vectorizer, test_texts, test_labels)
