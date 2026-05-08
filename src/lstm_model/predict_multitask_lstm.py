import torch
import torch.nn as nn
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

#define the model 

class MultiTaskLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_sent_classes, num_urg_classes):
        super(MultiTaskLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.fc_shared = nn.Linear(hidden_dim, 64)

        self.sent_head = nn.Linear(64, num_sent_classes)
        self.urg_head = nn.Linear(64, num_urg_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)

        x = hidden[-1]
        x = self.relu(self.fc_shared(x))

        sent_out = self.sent_head(x)
        urg_out = self.urg_head(x)

        return sent_out, urg_out


# check for cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and  encoders
with open("models/lstm/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/lstm/sent_encoder.pkl", "rb") as f:
    sent_encoder = pickle.load(f)

with open("models/lstm/urg_encoder.pkl", "rb") as f:
    urg_encoder = pickle.load(f)

# Model params should be equal to the traing 
vocab_size = 10000
embed_dim = 128
hidden_dim = 64
max_len = 100

num_sent_classes = len(sent_encoder.classes_)
num_urg_classes = len(urg_encoder.classes_)

# Load model
model = MultiTaskLSTM(vocab_size, embed_dim, hidden_dim, num_sent_classes, num_urg_classes)
model.load_state_dict(torch.load("models/lstm/multitask_lstm.pt", map_location=device))
model.to(device)
model.eval()


#predict 

def predict(text):
    # Tokenize
    sequence = tokenizer.texts_to_sequences([text])       #the tokenier
    padded = pad_sequences(sequence, maxlen=max_len, padding="post")  # padding the input 

    input_tensor = torch.tensor(padded, dtype=torch.long).to(device)

    with torch.no_grad():
        sent_out, urg_out = model(input_tensor)

        sent_pred = torch.argmax(sent_out, dim=1).cpu().numpy()
        urg_pred = torch.argmax(urg_out, dim=1).cpu().numpy()

    sentiment = sent_encoder.inverse_transform(sent_pred)[0]
    urgency = urg_encoder.inverse_transform(urg_pred)[0]

    return sentiment, urgency

#testing the model
if __name__ == "__main__":
    text = input("enter the review :")

    sentiment, urgency = predict(text)

    print("Text:", text)
    print("Sentiment:", sentiment)
    print("Urgency:", urgency)