from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def build_tokenizer(texts, vocab_size=20000, oov_token="<OOV>"):
    """
    Build and fit tokenizer on training texts.
    """

    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token=oov_token
    )

    tokenizer.fit_on_texts(texts)

    return tokenizer


def texts_to_padded_sequences(tokenizer, texts, max_len=200):
    """
    Convert texts → padded integer sequences.
    """

    sequences = tokenizer.texts_to_sequences(texts)

    padded = pad_sequences(
        sequences,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    return padded
    