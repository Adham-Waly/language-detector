import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())

    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))  # You can customize for other languages
    tokens = [token for token in tokens if token not in stop_words]

    # Join tokens back into text
    processed_text = ' '.join(tokens)

    return processed_text


def preprocess_data(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    languages = os.listdir(input_dir)
    for lang in languages:
        lang_dir = os.path.join(input_dir, lang)
        if not os.path.isdir(lang_dir):
            continue

        lang_files = os.listdir(lang_dir)

        output_lang_dir = os.path.join(output_dir, lang)
        if not os.path.exists(output_lang_dir):
            os.makedirs(output_lang_dir)

        for file_name in lang_files:
            input_file_path = os.path.join(lang_dir, file_name)
            output_file_path = os.path.join(output_lang_dir, file_name)
            try:
                with open(input_file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    processed_text = preprocess_text(text)

                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(processed_text)
            except Exception as e:
                print(f"Error processing file {input_file_path}: {e}")


if __name__ == "__main__":
    input_dir = r"A:\pythonProject\langue detection\data\raw_data"
    output_dir = r"A:\pythonProject\langue detection\data\processed_data"
    preprocess_data(input_dir, output_dir)
