import torch
import torch.nn as nn


# Attention Layer

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)   # (batch, seq_len, 1)
        context = torch.sum(weights * x, dim=1)         # (batch, hidden_dim*2)
        return context, weights

# Hybrid Model

class HybridModel(nn.Module):
    def __init__(self, embedding_matrix, tfidf_dim, hidden_dim, num_sent, num_urg):
        super().__init__()

        # Embedding Layer
        
        if embedding_matrix is not None:
            vocab_size, embed_dim = embedding_matrix.shape

            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))

        else:
            vocab_size = 10000
            embed_dim = 100

            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.embedding.weight.requires_grad = True

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=100,         # embed_dim
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3
        )

        # Attention
        self.attention = Attention(hidden_dim)

        # TF-IDF Projection
        self.tfidf_fc = nn.Linear(tfidf_dim, 128)

        # Fusion Layer
        self.fc = nn.Linear(hidden_dim * 2 + 128, 256)

        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

        
        # Output Heads
        self.sent_head = nn.Linear(256, num_sent)
        self.urg_head = nn.Linear(256, num_urg)

    # Forward
    def forward(self, x_seq, x_tfidf):

        # Embedding
        x = self.embedding(x_seq)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)

        # Attention
        attn_out, attn_weights = self.attention(lstm_out)

        # TF-IDF branch
        tfidf_out = self.relu(self.tfidf_fc(x_tfidf))

        # Fusion
        combined = torch.cat((attn_out, tfidf_out), dim=1)

        # Dense
        x = self.dropout(self.relu(self.fc(combined)))

        # Outputs
        sent_out = self.sent_head(x)
        urg_out = self.urg_head(x)

        return sent_out, urg_out, attn_weights