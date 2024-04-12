from constants import DATASET_PATH, SENTIMENT_MAPPING, TEMP_FILE_PATH
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

from util import preprocess_text_with_emojis

def predict_result(model, input_text):
    # Preprocess the dataset to fix formatting issues and load with correct column names
    with open(DATASET_PATH, 'r', encoding='utf-8') as file:
        data = file.read().splitlines(True)
    with open(TEMP_FILE_PATH, 'w', encoding='utf-8') as file:
        file.writelines(data[1:])  # Skip the first line containing headers

    # Load the preprocessed dataset using pd.read_csv() with correct column names
    dataset = pd.read_csv(TEMP_FILE_PATH, sep='\t', names=['text', 'category'])
    # Padding sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dataset['text'])
    sequences = tokenizer.texts_to_sequences(dataset['text'])
    max_len_text = max([len(seq) for seq in sequences])
    processed_text = preprocess_text_with_emojis(input_text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len_text, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction)
    for sentiment, label in SENTIMENT_MAPPING.items():
        if label == predicted_label:
            predicted_sentiment = sentiment
            break

    return predicted_sentiment