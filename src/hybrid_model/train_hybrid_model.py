# Hybrid Model: BiLSTM + Attention + TF-IDF + GloVe

import os
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn

from sklearn.utils import resample
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load Dataset
df = pd.read_csv("data/processed/playstore_clean.csv")

print("Before Balancing:")
print(df['sentiment'].value_counts())


# Balance Dataset
df_positive = df[df['sentiment'] == 'Positive']
df_negative = df[df['sentiment'] == 'Negative']
df_neutral  = df[df['sentiment'] == 'Neutral']

max_count = max(
    len(df_positive),
    len(df_negative),
    len(df_neutral)
)

df_negative_upsampled = resample(
    df_negative,
    replace=True,
    n_samples=max_count,
    random_state=42
)

df_neutral_upsampled = resample(
    df_neutral,
    replace=True,
    n_samples=max_count,
    random_state=42
)

df = pd.concat([
    df_positive,
    df_negative_upsampled,
    df_neutral_upsampled
])

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter Balancing:")
print(df['sentiment'].value_counts())


# Texts

texts = df["reviews"].astype(str).values


# Label Encoding
sent_encoder = LabelEncoder()
urg_encoder  = LabelEncoder()

sent_labels = sent_encoder.fit_transform(df["sentiment"])
urg_labels  = urg_encoder.fit_transform(df["urgency"])

print(f"Sentiment classes: {sent_encoder.classes_}")
print(f"Urgency classes: {urg_encoder.classes_}")


# TF-IDF
tfidf = TfidfVectorizer(max_features=8000)

X_tfidf = tfidf.fit_transform(texts).toarray()


# Manual Tokenization
vocab_size = 10000
max_len = 100

word_counts = Counter()

for text in texts:

    words = str(text).lower().split()

    word_counts.update(words)

# 0 = padding
# 1 = OOV
word_index = {
    word: idx + 2
    for idx, (word, _) in enumerate(
        word_counts.most_common(vocab_size - 2)
    )
}

print(f"Vocabulary Size: {len(word_index)}")



# Text → Sequence
def texts_to_sequences(texts, word_index):

    sequences = []

    for text in texts:

        words = str(text).lower().split()

        seq = [
            word_index.get(word, 1)
            for word in words
        ]

        sequences.append(seq)

    return sequences



# Manual Padding
def pad_sequence(seq, max_len):

    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))

    else:
        seq = seq[:max_len]

    return seq


# Generate Sequences
sequences = texts_to_sequences(texts, word_index)

X_seq = np.array([
    pad_sequence(seq, max_len)
    for seq in sequences
])

print(f"Sequence Shape: {X_seq.shape}")


# GloVe Embedding
def embedding_glove(glove_path, word_index, embed_dim=100):

    embeddings = {}

    with open(glove_path, encoding="utf-8") as f:

        for line in f:

            values = line.split()

            word = values[0]

            vec = np.asarray(
                values[1:],
                dtype="float32"
            )

            embeddings[word] = vec

    matrix = np.zeros(
        (len(word_index) + 2, embed_dim)
    )

    for word, i in word_index.items():

        vec = embeddings.get(word)

        if vec is not None:
            matrix[i] = vec

    return matrix


embedding_matrix = embedding_glove(
    "data/embedding/glove.6B.100d.txt",
    word_index
)

print(f"Embedding Matrix Shape: {embedding_matrix.shape}")


# Train/Test Split
(
    X_seq_train,
    X_seq_test,
    X_tfidf_train,
    X_tfidf_test,
    y_sent_train,
    y_sent_test,
    y_urg_train,
    y_urg_test

) = train_test_split(
    X_seq,
    X_tfidf,
    sent_labels,
    urg_labels,
    test_size=0.2,
    random_state=42
)



# Convert To Tensor
X_seq_train = torch.tensor(
    X_seq_train,
    dtype=torch.long
)

X_seq_test = torch.tensor(
    X_seq_test,
    dtype=torch.long
)

X_tfidf_train = torch.tensor(
    X_tfidf_train,
    dtype=torch.float32
)

X_tfidf_test = torch.tensor(
    X_tfidf_test,
    dtype=torch.float32
)

y_sent_train = torch.tensor(
    y_sent_train,
    dtype=torch.long
)

y_sent_test = torch.tensor(
    y_sent_test,
    dtype=torch.long
)

y_urg_train = torch.tensor(
    y_urg_train,
    dtype=torch.long
)

y_urg_test = torch.tensor(
    y_urg_test,
    dtype=torch.long
)


# Dataset
class HybridDataset(Dataset):

    def __init__(
        self,
        X_seq,
        X_tfidf,
        y_sent,
        y_urg
    ):

        self.X_seq = X_seq
        self.X_tfidf = X_tfidf
        self.y_sent = y_sent
        self.y_urg = y_urg

    def __len__(self):

        return len(self.X_seq)

    def __getitem__(self, idx):

        return (
            self.X_seq[idx],
            self.X_tfidf[idx],
            self.y_sent[idx],
            self.y_urg[idx]
        )


train_loader = DataLoader(
    HybridDataset(
        X_seq_train,
        X_tfidf_train,
        y_sent_train,
        y_urg_train
    ),
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    HybridDataset(
        X_seq_test,
        X_tfidf_test,
        y_sent_test,
        y_urg_test
    ),
    batch_size=64
)


# Attention Layer
class Attention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attn = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, x):

        weights = torch.softmax(
            self.attn(x),
            dim=1
        )

        context = torch.sum(
            weights * x,
            dim=1
        )

        return context, weights


# Hybrid Model
class HybridModel(nn.Module):

    def __init__(
        self,
        embedding_matrix,
        tfidf_dim,
        hidden_dim,
        num_sent,
        num_urg
    ):

        super().__init__()

        vocab_size, embed_dim = embedding_matrix.shape

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.embedding.weight.data.copy_(
            torch.tensor(embedding_matrix)
        )

        self.embedding.weight.requires_grad = True

        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3
        )

        self.attention = Attention(hidden_dim)

        self.tfidf_fc = nn.Linear(
            tfidf_dim,
            128
        )

        self.fc = nn.Linear(
            hidden_dim * 2 + 128,
            256
        )

        self.dropout = nn.Dropout(0.4)

        self.relu = nn.ReLU()

        self.sent_head = nn.Linear(
            256,
            num_sent
        )

        self.urg_head = nn.Linear(
            256,
            num_urg
        )

    def forward(self, x_seq, x_tfidf):

        x = self.embedding(x_seq)

        lstm_out, _ = self.bilstm(x)

        attn_out, attn_weights = self.attention(lstm_out)

        tfidf_out = self.relu(
            self.tfidf_fc(x_tfidf)
        )

        combined = torch.cat(
            (attn_out, tfidf_out),
            dim=1
        )

        x = self.dropout(
            self.relu(
                self.fc(combined)
            )
        )

        return (
            self.sent_head(x),
            self.urg_head(x),
            attn_weights
        )


# Model
model = HybridModel(
    embedding_matrix=embedding_matrix,
    tfidf_dim=X_tfidf.shape[1],
    hidden_dim=128,
    num_sent=len(sent_encoder.classes_),
    num_urg=len(urg_encoder.classes_)
).to(device)


criterion_sent = nn.CrossEntropyLoss()

criterion_urg = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
epochs = 12

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for (
        X_seq_b,
        X_tfidf_b,
        y_s,
        y_u

    ) in train_loader:

        X_seq_b = X_seq_b.to(device)

        X_tfidf_b = X_tfidf_b.to(device)

        y_s = y_s.to(device)

        y_u = y_u.to(device)

        optimizer.zero_grad()

        s_out, u_out, _ = model(
            X_seq_b,
            X_tfidf_b
        )

        loss = (
            criterion_sent(s_out, y_s)
            +
            criterion_urg(u_out, y_u)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {avg_loss:.4f}"
    )


# Save Model
os.makedirs(
    "models/hybrid",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    "models/hybrid/hybrid_model.pt"
)

with open(
    "models/hybrid/sent_encoder.pkl",
    "wb"
) as f:

    pickle.dump(sent_encoder, f)

with open(
    "models/hybrid/urg_encoder.pkl",
    "wb"
) as f:

    pickle.dump(urg_encoder, f)

with open(
    "models/hybrid/tfidf_vectorizer.pkl",
    "wb"
) as f:

    pickle.dump(tfidf, f)

with open(
    "models/hybrid/word_index.pkl",
    "wb"
) as f:

    pickle.dump(word_index, f)

print("Model and files saved successfully")