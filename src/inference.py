import pickle
import json

from tensorflow.keras.models import load_model
from utils.tokenizer import texts_to_padded_sequences
from preprocess import clean_text

from models.attention_layers import (
    SimpleAttention,
    BahdanauAttention,
    LuongAttention
)

class SentimentPredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)

        with open("saved_models/tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        with open("saved_models/config.json") as f:
            self.max_len = json.load(f)["max_len"]

    def predict(self, text):
        text = clean_text(text)
        seq = texts_to_padded_sequences(
            self.tokenizer,
            [text],
            self.max_len
        )
        prob = self.model.predict(seq, verbose=0)[0][0]

        if prob > 0.5:
            sentiment = "Positive"
        else:
            sentiment = "Negative"

        return sentiment, float(prob)
