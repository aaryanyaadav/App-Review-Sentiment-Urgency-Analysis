import pickle

# Loading the models and vectorizer
tfidf = pickle.load(open("models/baseline/tfidf_vectorizer.pkl", "rb"))
sentiment_model = pickle.load(open("models/baseline/sentiment_model.pkl", "rb"))
urgency_model = pickle.load(open("models/baseline/urgency_model.pkl", "rb"))

sentiment_encoder = pickle.load(open("models/baseline/sentiment_encoder.pkl", "rb"))
urgency_encoder = pickle.load(open("models/baseline/urgency_encoder.pkl", "rb"))

def predict(text):
    text_tfidf = tfidf.transform([text])

    sent_pred = sentiment_model.predict(text_tfidf)
    urg_pred = urgency_model.predict(text_tfidf)

    sentiment = sentiment_encoder.inverse_transform(sent_pred)[0]    #encode it back to text 
    urgency = urgency_encoder.inverse_transform(urg_pred)[0]         

    return sentiment, urgency

if __name__ == "__main__":
    text = input("enter the review:")


    sentiment, urgency = predict(text)

    print("Text:", text)
    print("Sentiment:", sentiment)
    print("Urgency:", urgency)