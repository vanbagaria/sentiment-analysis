from inference import SentimentPredictor

predictor = SentimentPredictor("saved_models/lstm_luong.keras")

input_line = ""
while True:
    print("Enter movie review text:")
    input_line = input()
    if(input_line == "exit"):
        break
    print("Sentiment: " + str(predictor.predict(input_line)))
