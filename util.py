# Function to identify and extract emojis from text
from constants import SENTIMENT_MAPPING
import emoji

def extract_emojis(text):
    return [char for char in text if char in emoji.UNICODE_EMOJI_ENGLISH]

# Function to map emojis to sentiment labels
def map_emojis_to_sentiment(emojis):
    return [SENTIMENT_MAPPING.get(emoji, 'Neutral') for emoji in emojis]

# Function to preprocess text and handle emojis
def preprocess_text_with_emojis(text):
    emojis = extract_emojis(text)
    emoji_sentiments = map_emojis_to_sentiment(emojis)
    processed_text = text
    for emoji, sentiment in zip(emojis, emoji_sentiments):
        processed_text = processed_text.replace(emoji, f"{sentiment} ")
    return processed_text