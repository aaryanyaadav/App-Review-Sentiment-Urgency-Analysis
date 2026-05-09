import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load the glove embeddings 
def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    embeddings_index = {}

    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        if i >= len(embedding_matrix):
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# load the data 

df = pd.read_csv("data/processed/playstore_clean.csv")

texts = df["reviews"].astype(str).values

# encode the labels
sent_encoder = LabelEncoder()
urg_encoder = LabelEncoder()

sent_labels = sent_encoder.fit_transform(df["sentiment"])
urg_labels = urg_encoder.fit_transform(df["urgency"])

# tokenization
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

max_len = 100
X = pad_sequences(sequences, maxlen=max_len, padding="post")

# golve embeddings

embedding_dim = 100
glove_path = "data/embedding/glove.6B.100d.txt"

embedding_matrix = load_glove_embeddings(glove_path, tokenizer.word_index, embedding_dim)

#test train split 

X_train, X_test, y_sent_train, y_sent_test, y_urg_train, y_urg_test = train_test_split(
    X, sent_labels, urg_labels, test_size=0.2, random_state=42
)

#convert to tensors
X_train = torch.tensor(X_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)

y_sent_train = torch.tensor(y_sent_train, dtype=torch.long)
y_sent_test = torch.tensor(y_sent_test, dtype=torch.long)

y_urg_train = torch.tensor(y_urg_train, dtype=torch.long)
y_urg_test = torch.tensor(y_urg_test, dtype=torch.long)

#dataset

class MultiTaskDataset(Dataset):
    def __init__(self, X, y_sent, y_urg):
        self.X = X
        self.y_sent = y_sent
        self.y_urg = y_urg

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_sent[idx], self.y_urg[idx]

train_loader = DataLoader(MultiTaskDataset(X_train, y_sent_train, y_urg_train), batch_size=64, shuffle=True)
test_loader = DataLoader(MultiTaskDataset(X_test, y_sent_test, y_urg_test), batch_size=64)

#attention layer added 
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context
    
#model 

class BiLSTMAttention(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_sent, num_urg):
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.embedding.weight.requires_grad = False  # freeze GloVe

        self.bilstm = nn.LSTM(
            embedding_dim, hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.attention = Attention(hidden_dim)

        self.fc = nn.Linear(hidden_dim * 2, 64)

        self.sent_head = nn.Linear(64, num_sent)
        self.urg_head = nn.Linear(64, num_urg)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.bilstm(x)

        attn_out = self.attention(lstm_out)

        x = self.relu(self.fc(attn_out))

        return self.sent_head(x), self.urg_head(x)

#initialization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiLSTMAttention(
    embedding_matrix,
    hidden_dim=64,
    num_sent=len(sent_encoder.classes_),
    num_urg=len(urg_encoder.classes_)
).to(device)

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

        loss = criterion(sent_out, y_sent_batch) + criterion(urg_out, y_urg_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")   


#evaluation
model.eval()
correct_sent = correct_urg = total = 0

with torch.no_grad():
    for X_batch, y_sent_batch, y_urg_batch in test_loader:
        X_batch = X_batch.to(device)

        sent_out, urg_out = model(X_batch)

        sent_pred = torch.argmax(sent_out, dim=1)
        urg_pred = torch.argmax(urg_out, dim=1)

        correct_sent += (sent_pred == y_sent_batch.to(device)).sum().item()
        correct_urg += (urg_pred == y_urg_batch.to(device)).sum().item()
        total += y_sent_batch.size(0)

print("Sentiment Accuracy:", correct_sent / total)
print("Urgency Accuracy:", correct_urg / total)

#saving the evaluation matrics

sent_acc = correct_sent / total
urg_acc = correct_urg / total

with open("outputs/reports/bilstm_attention_results.txt", "w") as f:
    f.write("BiLSTM + Attention Results:\n\n")
    f.write(f"Sentiment Accuracy: {sent_acc:.4f}\n")
    f.write(f"Urgency Accuracy: {urg_acc:.4f}\n")

print("evaluation score saved")

#saving the model and tokenizer

torch.save(model.state_dict(), "models/bilstm_attention_sen_urg/bilstm_attention.pt")

with open("models/bilstm_attention_sen_urg/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/bilstm_attention_sen_urg/sent_encoder.pkl", "wb") as f:
    pickle.dump(sent_encoder, f)

with open("models/bilstm_attention_sen_urg/urg_encoder.pkl", "wb") as f:
    pickle.dump(urg_encoder, f)

print(" Model and  tokenizer saved ")