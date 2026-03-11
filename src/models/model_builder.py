from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    GRU,
    Dense,
    Dropout,
    GlobalMaxPooling1D
)

from tensorflow.keras.models import Model

from models.attention_layers import (
    SimpleAttention,
    BahdanauAttention,
    LuongAttention
)

def build_model(
        vocab_size,
        max_len,
        rnn_type="lstm",
        attention=None,
        embedding_dim=128,
        rnn_units=128
):

    inputs = Input(shape=(max_len,))

    x = Embedding(
        vocab_size,
        embedding_dim,
        input_length=max_len
    )(inputs)

    # RNN layer
    if rnn_type == "lstm":
        x = LSTM(rnn_units, return_sequences=True)(x)

    elif rnn_type == "gru":
        x = GRU(rnn_units, return_sequences=True)(x)

    else:
        raise ValueError("Unsupported RNN type")

    # Attention layer
    if attention == "simple":
        x = SimpleAttention()(x)

    elif attention == "bahdanau":
        x = BahdanauAttention(rnn_units)(x)

    elif attention == "luong":
        x = LuongAttention()(x)

    else:
        x = GlobalMaxPooling1D()(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
