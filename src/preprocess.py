import re
import string
from nltk.corpus import stopwords

from tqdm import tqdm

# load stopwords once
STOPWORDS = set(stopwords.words("english"))


def clean_text(text, remove_stopwords=False):
    """
    Clean a single text review.
    """

    # lowercase
    text = text.lower()

    # remove HTML tags like <br />
    text = re.sub(r"<.*?>", " ", text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove numbers
    text = re.sub(r"\d+", "", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords:
        words = text.split()
        words = [w for w in words if w not in STOPWORDS]
        text = " ".join(words)

    return text

def preprocess_texts(texts, remove_stopwords=False):
    """
    Clean a list of texts.
    """

    cleaned = []

    for text in tqdm(texts, desc="Preprocessing text"):
        cleaned.append(clean_text(text, remove_stopwords))

    return cleaned