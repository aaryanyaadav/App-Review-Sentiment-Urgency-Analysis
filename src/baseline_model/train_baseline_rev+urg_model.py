import pandas as pd
import pickle 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
                                
df=pd.read_csv("data/processed/playstore_clean.csv")

sentiment_encoder = LabelEncoder()        
urgency_encoder = LabelEncoder()       

df["sentiment_label"] = sentiment_encoder.fit_transform(df["sentiment"])     #encoding the sentiment row and saving in sentiment label
df["urgency_label"] = urgency_encoder.fit_transform(df["urgency"])           #encoding the urgency row and saving in the urgeency label

#test train split 

#input
X = df["reviews"]
 
#output 

 
y_sentiment = df["sentiment_label"]
y_urgency = df["urgency_label"]

#sentiment 

X_train, X_test, y_sent_train, y_sent_test = train_test_split(
    X, y_sentiment, test_size=0.2, random_state=42
)

#urgency
_, _, y_urg_train, y_urg_test = train_test_split(
    X, y_urgency, test_size=0.2, random_state=42
)

# input transformation using the tfidf 
tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

#training model 

#initializing the model 

sentiment_model = LogisticRegression()
urgency_model = LogisticRegression()

#train 

sentiment_model.fit(X_train_tfidf, y_sent_train)
urgency_model.fit(X_train_tfidf, y_urg_train)

#print the

print("Sentiment Model:\n")
print(classification_report(y_sent_test, sentiment_model.predict(X_test_tfidf)))

print("Urgency Model:\n")
print(classification_report(y_urg_test, urgency_model.predict(X_test_tfidf)))

# saving the results
sent_report = classification_report(
    y_sent_test,
    sentiment_model.predict(X_test_tfidf)
)

urg_report = classification_report(
    y_urg_test,
    urgency_model.predict(X_test_tfidf)
)

with open("outputs/reports/baseline_rev+urg_results.txt", "w") as f:
    f.write("Sentiment Model\n\n")
    f.write(sent_report)
    
    f.write("\n\nUrgency Model \n\n")
    f.write(urg_report)

print(" Results saved to outputs/reports/baseline_rev+urg_results.txt")

# save vectorizer 
with open("models/baseline/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

#save model

with open("models/baseline/sentiment_model.pkl", "wb") as f:
    pickle.dump(sentiment_model, f)

with open("models/baseline/urgency_model.pkl", "wb") as f:
    pickle.dump(urgency_model, f)

#save the endoded values

with open("models/baseline/sentiment_encoder.pkl", "wb") as f:
    pickle.dump(sentiment_encoder, f)

with open("models/baseline/urgency_encoder.pkl", "wb") as f:
    pickle.dump(urgency_encoder, f)

print(" Models and vectorizer saved successfully!")