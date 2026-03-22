import streamlit as st
from inference import SentimentPredictor

st.title("Movie Review Sentiment Analyzer")

# Model selection
rnn_type = st.selectbox("Select RNN Type:", ["lstm", "gru"])
attention_type = st.selectbox("Select Attention Mechanism:", ["none", "simple", "bahdanau", "luong"])

# Caching the model based on user selection
@st.cache_resource
def get_model(rnn, attention):
    model_name = f"{rnn}_{attention}"
    return SentimentPredictor(f"saved_models/{model_name}.keras")

predictor = get_model(rnn_type, attention_type)
input_text = st.text_input("Enter movie review:", key="review_input")

if st.button("Analyze") or (input_text and st.session_state.review_input):
    if input_text:
        sentiment = predictor.predict(input_text)
        if sentiment[0] == "Positive":
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text first!")
