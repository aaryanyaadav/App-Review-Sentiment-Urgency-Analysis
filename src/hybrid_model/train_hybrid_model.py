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

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# check if gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#load the data

df = pd.read_csv("data/processed/playstore_clean.csv")

# oversampling the majority classes

print("Before Balancing:")
print(df['sentiment'].value_counts())

#separate rows by sentiment class
df_positive = df[df['sentiment'] == 'Positive']
df_negative = df[df['sentiment'] == 'Negative']
df_neutral  = df[df['sentiment'] == 'Neutral']

#finding the largest sentiment
max_count = max(len(df_positive), len(df_negative), len(df_neutral))

df_neutral_upsampled = resample(
    df_neutral,
    replace=True,
    n_samples=max_count,
    random_state=42
)
df_negative_upsampled = resample(
    df_negative,
    replace=True,
    n_samples=max_count,
    random_state=42
)

#combine all three balanced classes and shuffle the rows
df = pd.concat([df_positive, df_negative_upsampled, df_neutral_upsampled])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter Balancing:")
print(df['sentiment'].value_counts())


texts = df["reviews"].astype(str).values

#label encode the sentiment and urgency

sent_encoder = LabelEncoder()
urg_encoder  = LabelEncoder()

sent_labels = sent_encoder.fit_transform(df["sentiment"])
urg_labels  = urg_encoder.fit_transform(df["urgency"])

print(f"Sentiment classes: {sent_encoder.classes_}")
print(f"Urgency classes:   {urg_encoder.classes_}")

#TF-IDF features


tfidf = TfidfVectorizer(max_features=8000)
X_tfidf = tfidf.fit_transform(texts).toarray()  


#Tokenization

vocab_size = 10000
tokenizer  = Tokenizer(num_words=vocab_size, oov_token="<OOV>") 
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

#padding the reviews 

max_len = 100
X_seq = pad_sequences(sequences, maxlen=max_len, padding="post")   # shape: (num_reviews, 100)

#glove embedding
def embedding_glove(glove_path, word_index, embed_dim=100):
    embeddings = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec  = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vec

    matrix = np.zeros((len(word_index) + 1, embed_dim))

    for word, i in word_index.items():
        vec = embeddings.get(word)
        if vec is not None:
            matrix[i] = vec  

    return matrix

embedding_matrix = embedding_glove(
    "data/embedding/glove.6B.100d.txt",
    tokenizer.word_index
)
print(f"Embedding matrix shape: {embedding_matrix.shape}")


#Test-train split

(X_seq_train, X_seq_test,
 X_tfidf_train, X_tfidf_test,
 y_sent_train, y_sent_test,
 y_urg_train, y_urg_test) = train_test_split(
    X_seq, X_tfidf, sent_labels, urg_labels,
    test_size=0.2,
    random_state=42
)

#converting into pytorch tensors 
X_seq_train   = torch.tensor(X_seq_train,   dtype=torch.long)
X_seq_test    = torch.tensor(X_seq_test,    dtype=torch.long)

X_tfidf_train = torch.tensor(X_tfidf_train, dtype=torch.float32)
X_tfidf_test  = torch.tensor(X_tfidf_test,  dtype=torch.float32)

y_sent_train  = torch.tensor(y_sent_train,  dtype=torch.long)
y_sent_test   = torch.tensor(y_sent_test,   dtype=torch.long)

y_urg_train   = torch.tensor(y_urg_train,   dtype=torch.long)
y_urg_test    = torch.tensor(y_urg_test,    dtype=torch.long)


#dataset
class HybridDataset(Dataset):
    def __init__(self, X_seq, X_tfidf, y_sent, y_urg):
        self.X_seq   = X_seq
        self.X_tfidf = X_tfidf
        self.y_sent  = y_sent
        self.y_urg   = y_urg

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_tfidf[idx], self.y_sent[idx], self.y_urg[idx]


train_loader = DataLoader(
    HybridDataset(X_seq_train, X_tfidf_train, y_sent_train, y_urg_train),
    batch_size=64,
    shuffle=True
)
test_loader = DataLoader(
    HybridDataset(X_seq_test, X_tfidf_test, y_sent_test, y_urg_test),
    batch_size=64
)


#attention 

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(weights * x, dim=1)        # (batch, hidden_dim*2)
        return context, weights              


#Hybrid Model
#   BiLSTM + Attention → understands word order and context
#   TF-IDF  → understands word frequency


class HybridModel(nn.Module):
    def __init__(self, embedding_matrix, tfidf_dim, hidden_dim, num_sent, num_urg):
        super().__init__()

        vocab_size, embed_dim = embedding_matrix.shape

        # embedding layer (numbers to glove vecores)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.embedding.weight.requires_grad = True

        #BiLSTM Layer 
        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2, #two lstm
            dropout=0.3  #turn off 30 neuorns to prevent overfitting
        )

        # Attention layer
        self.attention = Attention(hidden_dim)

        #TF-IDF Projection Layer (8000 dim to 128 dims)
        self.tfidf_fc = nn.Linear(tfidf_dim, 128)

        # Fusion Layer (bilstm + tf-idf)
        self.fc = nn.Linear(hidden_dim * 2 + 128, 256)

        self.dropout = nn.Dropout(0.4)   # 40% dropout to prevent overfitting
        self.relu    = nn.ReLU()         # activation function (adds non-linearity)

        # Output 
        self.sent_head = nn.Linear(256, num_sent)   # sentiment class
        self.urg_head  = nn.Linear(256, num_urg)    # urgency class

    def forward(self, x_seq, x_tfidf):
        # Convert word IDs → GloVe vectors
        x = self.embedding(x_seq)                   

        # BiLSTM 
        lstm_out, _ = self.bilstm(x)           

        # Attention 
        attn_out, attn_weights = self.attention(lstm_out)  

        # TF-IDF 
        tfidf_out = self.relu(self.tfidf_fc(x_tfidf))  # (batch, 128)

        # Fusing both of the outputs
        combined = torch.cat((attn_out, tfidf_out), dim=1)  

        # dense layer and output
        x = self.dropout(self.relu(self.fc(combined)))        # (batch, 256)

        # Two separate predictions
        return self.sent_head(x), self.urg_head(x) ,attn_weights 


model = HybridModel(
    embedding_matrix=embedding_matrix,
    tfidf_dim=X_tfidf.shape[1],         # 8000
    hidden_dim=128,
    num_sent=len(sent_encoder.classes_),
    num_urg=len(urg_encoder.classes_)
).to(device)

# CrossEntropyLoss
criterion_sent = nn.CrossEntropyLoss()
criterion_urg  = nn.CrossEntropyLoss()

# Adam optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# training loop

epochs = 12

for epoch in range(epochs):
    model.train()    # set model to training mode (enables dropout etc.)
    total_loss = 0

    for X_seq_b, X_tfidf_b, y_s, y_u in train_loader:
        X_seq_b   = X_seq_b.to(device)
        X_tfidf_b = X_tfidf_b.to(device)
        y_s       = y_s.to(device)
        y_u       = y_u.to(device)

        # Clear gradients from previous step
        optimizer.zero_grad()

        # Forward pass —
        s_out, u_out, _ = model(X_seq_b, X_tfidf_b)

        # Calculate the combined loss (sentiment + urgency)
    
        loss = criterion_sent(s_out, y_s) + criterion_urg(u_out, y_u)

        # Backward pass (gradients)
        loss.backward()

        # Update model weights based on gradients
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}]  Loss: {total_loss:.4f}  Avg Loss: {avg_loss:.4f}")


# Evaluation

model.eval()   

all_sent_preds = []
all_sent_true  = []
all_urg_preds  = []
all_urg_true   = []

with torch.no_grad():   # no gradient calculation needed during evaluation
    for X_seq_b, X_tfidf_b, y_s, y_u in test_loader:
        X_seq_b   = X_seq_b.to(device)
        X_tfidf_b = X_tfidf_b.to(device)

        s_out, u_out,_ = model(X_seq_b, X_tfidf_b)
        s_pred = torch.argmax(s_out, dim=1).cpu().numpy()
        u_pred = torch.argmax(u_out, dim=1).cpu().numpy()

        all_sent_preds.extend(s_pred)
        all_sent_true.extend(y_s.numpy())
        all_urg_preds.extend(u_pred)
        all_urg_true.extend(y_u.numpy())

#sentiment
print("Sentiment:")
print(classification_report(
    all_sent_true,
    all_sent_preds,
    target_names=sent_encoder.classes_
))

#urgency
print("Urgency:")
print(classification_report(
    all_urg_true,
    all_urg_preds,
    target_names=urg_encoder.classes_
))



#Saving the model and tokenizer and vectorizer

os.makedirs("models", exist_ok=True)

# Save the trained model
torch.save(model.state_dict(), "models/hybrid/hybrid_model.pt")

# Save encoders and vectorizers
with open("models/hybrid/sent_encoder.pkl", "wb") as f:
    pickle.dump(sent_encoder, f)

with open("models/hybrid/urg_encoder.pkl", "wb") as f:
    pickle.dump(urg_encoder, f)

with open("models/hybrid/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("models/hybrid/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("model & other  saved")