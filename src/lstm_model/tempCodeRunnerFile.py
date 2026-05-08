import  pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load the dataset

df = pd.read_csv("data/processed/playstore_clean.csv")

texts = df["reviews"].astype(str).values

# Encoding the labels
sent_encoder = LabelEncoder()
urg_encoder = LabelEncoder()

sent_labels = sent_encoder.fit_transform(df["sentiment"])
urg_labels = urg_encoder.fit_transform(df["urgency"])

#tokenization

vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

# Padding
#make all the review of size 100 with the help of padding 
max_len = 50
X = pad_sequences(sequences, maxlen=max_len, padding="post")

# Train-test split
X_train, X_test, y_sent_train, y_sent_test, y_urg_train, y_urg_test = train_test_split(
    X, sent_labels, urg_labels, test_size=0.2, random_state=42
)

#covert to tensors 

X_train = torch.tensor(X_train, dtype=torch.long)
y_sent_train = torch.tensor(y_sent_train, dtype=torch.long)
y_urg_train = torch.tensor(y_urg_train, dtype=torch.long)

#test data 

X_test = torch.tensor(X_test, dtype=torch.long)
y_sent_test = torch.tensor(y_sent_test, dtype=torch.long)
y_urg_test = torch.tensor(y_urg_test, dtype=torch.long)

#dataset
#create a custom dataset for training 
class MultiTaskDataset(Dataset):
    def __init__(self, X, y_sent, y_urg):
        self.X = X
        self.y_sent = y_sent
        self.y_urg = y_urg

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_sent[idx], self.y_urg[idx]
    
#create dataloaders 

train_dataset = MultiTaskDataset(X_train, y_sent_train, y_urg_train)
test_dataset = MultiTaskDataset(X_test, y_sent_test, y_urg_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# model training
class MultiTaskLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(MultiTaskLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Shared layer
        self.fc_shared = nn.Linear(hidden_dim, 64)

        # Task-specific heads
        self.sent_head = nn.Linear(64, 3)
        self.urg_head = nn.Linear(64, 3)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)

        x = hidden[-1]
        x = self.relu(self.fc_shared(x))

        sent_out = self.sent_head(x)
        urg_out = self.urg_head(x)

        return sent_out, urg_out

# initialize 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiTaskLSTM(vocab_size, 128, 64).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#training

epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_sent_batch, y_urg_batch in train_loader:
        X_batch = X_batch.to(device)
        y_sent_batch = y_sent_batch.to(device)
        y_urg_batch = y_urg_batch.to(device)

        optimizer.zero_grad()

        sent_out, urg_out = model(X_batch)

        loss_sent = criterion(sent_out, y_sent_batch)
        loss_urg = criterion(urg_out, y_urg_batch)

        loss = loss_sent + loss_urg  # combine losses

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# evaluation 

model.eval()

correct_sent, correct_urg, total = 0, 0, 0

with torch.no_grad():
    for X_batch, y_sent_batch, y_urg_batch in test_loader:
        X_batch = X_batch.to(device)
        y_sent_batch = y_sent_batch.to(device)
        y_urg_batch = y_urg_batch.to(device)

        sent_out, urg_out = model(X_batch)

        _, sent_pred = torch.max(sent_out, 1)
        _, urg_pred = torch.max(urg_out, 1)

        total += y_sent_batch.size(0)

        correct_sent += (sent_pred == y_sent_batch).sum().item()
        correct_urg += (urg_pred == y_urg_batch).sum().item()

print("Sentiment Accuracy:", correct_sent / total)
print("Urgency Accuracy:", correct_urg / total)

#save the model and tokenizer

torch.save(model.state_dict(), "models/lstm/multitask_lstm.pt")

with open("models/lstm/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/lstm/sent_encoder.pkl", "wb") as f:
    pickle.dump(sent_encoder, f)

with open("models/lstm/urg_encoder.pkl", "wb") as f:
    pickle.dump(urg_encoder, f)

print("Model and vectorizer saved")

#saving the results

sent_accuracy = correct_sent / total
urg_accuracy = correct_urg / total

with open("outputs/reports/multitask_lstm_results.txt", "w") as f:
    f.write("Multi-Task LSTM Results:\n\n")
    f.write(f"Sentiment Accuracy: {sent_accuracy:.4f}\n")
    f.write(f"Urgency Accuracy: {urg_accuracy:.4f}\n")

print("Results saved")